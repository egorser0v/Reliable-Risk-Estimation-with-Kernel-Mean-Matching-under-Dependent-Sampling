"""Two real-data estimands with a fixed predictor and source-only uncertainty.

Run with Python 3.8 + numpy/pandas/scipy/sklearn/h5py:
  python experiments/kmm_three_domain_benchmark.py --dataset all
Outputs are new and do not overwrite historical experiments.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work/kmm_py38'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

SEED = 20261005
OUT = ROOT / 'results/kmm_three_domain'
NOAA = Path(os.environ.get('KMM_NOAA_ROOT', r'abc'))
B = 10.


def project(v):
    lo, hi = v.min()-B, v.max()
    for _ in range(45):
        mid = (lo+hi)/2
        if np.clip(v-mid, 0, B).sum() > len(v): lo = mid
        else: hi = mid
    return np.clip(v-(lo+hi)/2, 0, B)


def weights(xs, xt, verify=False):
    # One scaler and one bandwidth shared by every Gram matrix.
    prep = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    pooled = prep.fit_transform(np.vstack([xs, xt]))
    xs, xt = pooled[:len(xs)], pooled[len(xs):]
    d = pdist(pooled)
    sigma2 = max(float(np.median(d[d>0])**2), 1e-8)
    K = np.exp(-cdist(xs,xs,'sqeuclidean')/(2*sigma2))
    cross = np.exp(-cdist(xs,xt,'sqeuclidean')/(2*sigma2))
    target = np.exp(-cdist(xt,xt,'sqeuclidean')/(2*sigma2)).mean()
    A = K + np.eye(len(xs))*1e-8
    b = len(xs)*cross.mean(1)
    L = A.sum(1).max()
    w = np.ones(len(xs)); z = w.copy(); t = 1.
    # Small temporal QPs converge faster with SLSQP; retain FISTA for the
    # independent first-task comparison and larger spatial windows.
    use_direct = len(xs)<150 and not verify
    for iteration in range(0 if use_direct else 6000):
        nxt = project(z-(A@z-b)/L)
        tn = (1+np.sqrt(1+4*t*t))/2
        z = nxt+(t-1)/tn*(nxt-w)
        w,t = nxt,tn
        if iteration % 50 == 0:
            residual = np.linalg.norm(w-project(w-(A@w-b)/L)) / max(np.linalg.norm(w),1)
            if residual < 1e-7: break
    residual = np.linalg.norm(w-project(w-(A@w-b)/L))/max(np.linalg.norm(w),1)
    # Independent solver check; if FISTA needs refinement, solve the same QP.
    obj = lambda a: .5*a@A@a-b@a
    solver_difference = np.nan
    if use_direct: iteration=-1
    if use_direct or verify or residual > 2e-6:
        opt = minimize(obj,w,jac=lambda a:A@a-b,method='SLSQP',bounds=[(0,B)]*len(w),
                       constraints=[{'type':'eq','fun':lambda a:a.sum()-len(a),'jac':lambda a:np.ones(len(a))}],
                       options={'maxiter':2500,'ftol':1e-8})
        if not opt.success: raise RuntimeError('KMM solver: '+str(opt.message))
        solver_difference = np.nan if use_direct else abs(obj(w)-obj(opt.x))/max(1,abs(obj(opt.x)))
        if obj(opt.x)<obj(w): w=opt.x
    residual = np.linalg.norm(w-project(w-(A@w-b)/L))/max(np.linalg.norm(w),1)
    assert abs(w.sum()-len(w)) < 1e-5 and w.min()>-1e-6 and w.max()<B+1e-6
    clf = LogisticRegression(C=1.,max_iter=1000,random_state=SEED).fit(pooled,np.r_[np.zeros(len(xs)),np.ones(len(xt))])
    prob = np.clip(clf.predict_proba(xs)[:,1],1e-5,1-1e-5)
    ratio = np.minimum(prob/(1-prob)*len(xs)/len(xt),B)
    allw = {'Unweighted':np.ones(len(xs)), 'Logistic IW':ratio, 'KMM':w}
    def mmd(a):
        a=a/a.sum()
        return max(0.,float(a@K@a+target-2*a@cross.mean(1)))
    info = {'bandwidth':sigma2**.5,'mmd2_before':mmd(np.ones(len(xs))),
            'kmm_projected_residual':float(residual),'solver_objective_relative_difference':solver_difference,
            'kmm_iterations':iteration+1}
    return allw, mmd, info


def correction_matrix(frame, spatial):
    if spatial:
        xy=frame[['coord_x','coord_y']].to_numpy()
        dist=cdist(xy,xy)
        nearest=np.partition(dist,1,axis=1)[:,1]
        bandwidth=max(5*float(np.median(nearest)),1e-6)
        # Gaussian covariance taper is PSD; coordinates never cross images.
        return np.exp(-.5*(dist/bandwidth)**2), bandwidth
    days=(frame.date-frame.date.min()).dt.days.to_numpy()
    # Bartlett HAC at actual calendar lags, never concatenating different years.
    return np.maximum(0,1-abs(days[:,None]-days[None,:])/15.),14.


def evaluate(name,task,src,tgt,features,model,extra,verify):
    xs,xt=src[features].to_numpy(),tgt[features].to_numpy()
    ws,mmd,info=weights(xs,xt,verify)
    corr,lag=correction_matrix(src,name=='TNBC')
    # Predictions, weighting, and covariance geometry do not access target outcomes.
    ps,pt=model.predict(xs),model.predict(xt)
    rows=[]
    for estimand in ['mean','risk']:
        ys=src.response.to_numpy()
        if estimand=='risk': ys=(np.clip(ps,0,1)-ys)**2
        # Only this evaluation step reveals held-out target labels.
        yt=tgt.response.to_numpy()
        if estimand=='risk': yt=(np.clip(pt,0,1)-yt)**2
        truth=float(yt.mean())
        for method,w in ws.items():
            wn=w/w.sum(); estimate=float(wn@ys)
            z=wn*(ys-estimate)
            viid=float(z@z)*len(ys)/(len(ys)-1)
            vdep=float(z@corr@z)*len(ys)/(len(ys)-1)
            ess=float(w.sum()**2/(w@w))
            for adjustment in (['iid','dependence'] if method=='KMM' else ['iid']):
                var=viid if adjustment=='iid' else max(viid,vdep)
                hw=1.96*np.sqrt(max(0,var))
                adjusted=ess if adjustment=='iid' else max(1.,ess*viid/max(var,1e-20))
                rows.append(dict(dataset=name,task_id=task,estimand=estimand,
                    method=method+(' + dependence' if adjustment=='dependence' else ''),
                    estimate=estimate,true_target=truth,absolute_error=abs(estimate-truth),
                    coverage=int(abs(estimate-truth)<=hw),half_width=hw,
                    nominal_ess=ess,adjusted_ess=adjusted,max_weight=float(w.max()),
                    sum_squared_weights=float(w@w),zero_weight_fraction=float(np.mean(w<1e-6)),
                    upper_bound_fraction=float(np.mean(w>B-1e-5)),
                    mmd2_after=mmd(w),dependence_variance_ratio=vdep/max(viid,1e-20),
                    dependence_bandwidth=lag,n_source=len(src),n_target=len(tgt),
                    source_domain=str(src.domain.iloc[0]),target_domain=str(tgt.domain.iloc[0]),
                    **info,**extra))
    return rows


def load_pm():
    df=pd.read_csv(ROOT/'data/pm25/beijing_pm25_daily_prepared.csv',parse_dates=['date'])
    df['domain']=df.station
    df['response']=np.clip(df['PM2.5']/500.,0,1)
    features=['TEMP','PRES','DEWP','RAIN','WSPM','SO2','NO2','CO','O3','day_of_year_sin','day_of_year_cos']
    df=df.dropna(subset=['response'])
    train=df[df.date.dt.year<2015]
    tasks=[]
    for station,g in df.groupby('station'):
        for quarter in range(1,5):
            src=g[(g.date.dt.year==2015)&(g.date.dt.quarter==quarter)].sort_values('date')
            tgt=g[(g.date.dt.year==2016)&(g.date.dt.quarter==quarter)].sort_values('date')
            if min(len(src),len(tgt))>=65:
                tasks.append((f'{station}_Q{quarter}',src,tgt,{'regime':'temporal_2015_2016','replicate':0}))
    return train,features,tasks,{'response':'PM2.5 clipped at 500 ug/m3, divided by 500','training':'2013-2014','n_rows':len(df)}


def load_noaa():
    cache=ROOT/'data/noaa/ghcn2025_benchmark.csv'
    if cache.exists(): df=pd.read_csv(cache,parse_dates=['date'])
    else:
        cov=pd.read_csv(NOAA/'coverage.csv')
        ids=cov[(cov.complete>=330)&(cov.prcp>=300)].station
        meta=pd.read_csv(NOAA/'candidate_metadata.csv').set_index('station')
        frames=[]
        for sid in ids:
            d=pd.read_csv(NOAA/f'{sid}.csv.gz',header=None,names=['station','date','element','value','mflag','qflag','sflag','obstime'],dtype={'date':str},low_memory=False)
            d=d[(d.date.str[:4]=='2025')&d.element.isin(['TMAX','TMIN','PRCP'])&d.qflag.isna()]
            g=d.pivot_table(index='date',columns='element',values='value',aggfunc='first').reset_index()
            g=g.dropna(subset=['TMAX','TMIN','PRCP'])
            for col in ['TMAX','TMIN','PRCP']: g[col]=g[col]/10
            g['date']=pd.to_datetime(g.date); g['domain']=sid
            g['latitude']=meta.loc[sid,'lat']; g['longitude']=meta.loc[sid,'lon']
            frames.append(g)
        df=pd.concat(frames,ignore_index=True)
        cache.parent.mkdir(parents=True,exist_ok=True); df.to_csv(cache,index=False)
    df['response']=np.clip((df.TMAX+30)/80,0,1)
    df['sin_doy']=np.sin(2*np.pi*df.date.dt.dayofyear/365.25)
    df['cos_doy']=np.cos(2*np.pi*df.date.dt.dayofyear/365.25)
    features=['TMIN','PRCP','sin_doy','cos_doy']
    ids=np.array(sorted(df.domain.unique())); np.random.default_rng(SEED).shuffle(ids)
    train_ids=ids[:20]; targets=ids[20:32]; sources=ids[32:]
    meta=df.groupby('domain')[['latitude','longitude']].first()
    tasks=[]
    for tid in targets:
        lat,lon=meta.loc[tid]
        slat=np.radians(meta.loc[sources,'latitude'].to_numpy())
        slon=np.radians(meta.loc[sources,'longitude'].to_numpy())
        a=np.sin((slat-np.radians(lat))/2)**2+np.cos(slat)*np.cos(np.radians(lat))*np.sin((slon-np.radians(lon))/2)**2
        distances=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
        for regime,desired in [('mild',150),('medium',600),('severe',1200)]:
            j=np.argmin(abs(distances-desired)); sid=sources[j]
            for q in range(1,5):
                src=df[(df.domain==sid)&(df.date.dt.quarter==q)].sort_values('date')
                tgt=df[(df.domain==tid)&(df.date.dt.quarter==q)].sort_values('date')
                if min(len(src),len(tgt))>=65:
                    tasks.append((f'{tid}_{regime}_Q{q}',src,tgt,{'regime':regime,'replicate':0,'distance_km':distances[j]}))
    return df[df.domain.isin(train_ids)],features,tasks,{'response':'TMAX clipped to [-30,50] C, normalized','training_stations':train_ids.tolist(),'n_rows':len(df)}


def load_tnbc():
    import h5py
    with h5py.File(ROOT/'data/tnbc/tnbc.h5ad','r') as f:
        cols=[s.decode() for s in f['var']['_index'][:]]
        df=pd.DataFrame(f['X'][:],columns=cols)
        df['domain']=f['obs']['SampleID'][:]
        df['coord_x']=f['obs']['x'][:]; df['coord_y']=f['obs']['y'][:]
    ids=np.array(sorted(df.domain.unique())); np.random.default_rng(SEED).shuffle(ids)
    train_ids=ids[:8]; eval_ids=ids[8:]
    # Continuous marker endpoint avoids using phenotypes derived from all input markers.
    scale=float(df[df.domain.isin(train_ids)].Ki67.quantile(.99))
    if not np.isfinite(scale) or scale<=0: raise ValueError('Invalid Ki67 scale')
    df['response']=np.clip(df.Ki67/scale,0,1)
    features=[c for c in cols if c!='Ki67']
    tasks=[]
    def window(g,r):
        # Connected local neighborhoods, selected from coordinates without labels.
        xy=g[['coord_x','coord_y']].to_numpy()
        center=np.quantile(xy,[.25,.5,.75][r],axis=0)
        order=np.argsort(((xy-center)**2).sum(1),kind='stable')[:256]
        return g.iloc[order]
    for j,tid in enumerate(eval_ids):
        sid=eval_ids[(j+1)%len(eval_ids)]
        for r in range(3):
            src=window(df[df.domain==sid],r); tgt=window(df[df.domain==tid],r)
            tasks.append((f'patient{tid}_window{r}',src,tgt,{'regime':'patient_transfer','replicate':r}))
    return df[df.domain.isin(train_ids)],features,tasks,{'response':'Positive part of processed standardized Ki67, clipped at training-patient 99th percentile and normalized; not raw concentration','response_scale':scale,'training_patients':train_ids.tolist(),'n_rows':len(df),'n_patients':len(ids)}


def tables():
    files=sorted(OUT.glob('*_tasks.csv'))
    if not files: return
    rows=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    summaries=[]; latex_tables=[]
    for estimand in ['mean','risk']:
        d=rows[rows.estimand==estimand]
        summary=d.groupby(['dataset','method'],sort=False).agg(MAE=('absolute_error','mean'),Coverage=('coverage','mean'),Half_width=('half_width','mean'),ESS=('adjusted_ess','mean'),Tasks=('task_id','size'),Domains=('target_domain','nunique')).reset_index()
        summary.to_csv(OUT/f'{estimand}_summary.csv',index=False)
        caption=('Target-mean' if estimand=='mean' else 'Fixed-predictor target-risk')+' estimation. Outcomes are bounded in [0,1]; risk is squared error of a separately trained, frozen random forest. Intervals are nominal 95% source-only plug-in intervals, conditional on fitted weights and target covariates. Coverage is a descriptive finite-target benchmark, not theorem coverage. Repeated tasks share domains.'
        tex=summary.to_latex(index=False,float_format=lambda x:f'{x:.4f}',caption=caption,label=f'tab:real_{estimand}',position='t')
        (OUT/f'{estimand}_table.tex').write_text(tex,encoding='utf8')
        latex_tables.append(tex)
        summaries.append((estimand,summary))
    lines=['# Three-domain real-data benchmark','',
        'Two estimands: bounded target response mean and target squared-error risk of a frozen random forest. All target outcomes are used only for scoring. No target labels select weights, predictors, windows, kernel bandwidth, or uncertainty parameters.',
        '', '## Tables (normalized units)']
    for label,s in summaries:
        lines += ['', '### '+label,'',s.to_string(index=False)]
    lines += ['', '## Results by dataset','']
    for label,s in summaries:
        for dataset,g in s.groupby('dataset'):
            v=g.set_index('method'); u=v.loc['Unweighted']; k=v.loc['KMM']; a=v.loc['KMM + dependence']
            improvement=100*(u.MAE-k.MAE)/u.MAE
            lines += [f'{dataset}, {label}: KMM MAE change versus unweighted = {improvement:+.1f}% improvement; KMM iid coverage {100*k.Coverage:.1f}% -> dependence correction {100*a.Coverage:.1f}%; half-width {k.Half_width:.6f} -> {a.Half_width:.6f}.']
    lines += ['', 'Positive percentages above mean lower point-estimation error; negative percentages mean deterioration. These descriptive differences are not significance tests. The experiment does not establish universal point-estimation superiority or nominal calibration.']
    lines += ['', '## Interpretation and limitations','',
        'These are empirical stress tests, not finite-sample theorem verification. The intervals estimate source variation conditional on fitted weights and fixed target covariates; weight-fitting and target-population sampling uncertainty are not fully represented. The reference is the observed finite target-window mean/risk, not an independently known population expectation.',
        'Source temporal covariance uses Bartlett HAC at actual calendar lags through 14 days. TNBC uses a PSD Gaussian spatial covariance taper with bandwidth five times the median within-window nearest-neighbor distance. Corrected variance is floored at iid variance; this is an explicit empirical conservative correction, not the GP-derived ESS formula.',
        'Tasks share stations/patients and windows; task counts are not independent replicate counts. Any inference across tasks should resample whole domains. No selection based on coverage or target errors is performed.',
        'TNBC is a processed public secondary release of Keren et al. data (https://doi.org/10.6084/m9.figshare.26068006.v2). The X matrix is already standardized and contains negative marker values. The endpoint is explicitly the positive part of processed Ki67, upper-clipped at the training-patient 99th percentile and divided by that percentile; it is not raw concentration or clinical proliferation risk. Ki67 is excluded from predictors; derived cell-type/niche labels and spatial embeddings are excluded. Upstream cohort-level preprocessing was not refit and may carry batch effects or use held-out distributions; source-only preprocessing claims refer to this experiment, not the original release. Predictor training patients are disjoint from every evaluation patient.',
        'NOAA compares individual station-day sequences, with training stations disjoint from source and target stations; source and target station pools are disjoint. PM2.5 trains on 2013-2014 and evaluates 2015 -> 2016 by station and quarter. Neither naturally occurring shift guarantees conditional invariance.',
        'One pooled covariate-only scaler and median bandwidth are used for all KMM Gram matrices. Weights obey 0<=w<=10 and sum(w)=n. Logistic ratios are upper-capped and self-normalized for estimation. Zero weights and upper-bound weights are reported separately.',
        'Historical PM2.5/NOAA numbers are not pooled with this corrected protocol. This run replaces neither historical files nor manuscript text. See task CSVs and dataset protocol JSON for complete diagnostics and provenance.',
        '', '## Reproduction','', '`python experiments/kmm_three_domain_benchmark.py --dataset all`',
        'Set KMM_NOAA_ROOT to the directory containing coverage.csv, candidate_metadata.csv and station gzip files if rebuilding the NOAA cache. Run prepare_tnbc_benchmark.py to download TNBC. Requirements: numpy, pandas, scipy, scikit-learn, threadpoolctl, h5py. Seed: '+str(SEED)]
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8')
    diagnostic_cols=['mmd2_before','mmd2_after','nominal_ess','adjusted_ess','max_weight','sum_squared_weights','zero_weight_fraction','upper_bound_fraction','dependence_variance_ratio']
    rows.groupby(['dataset','estimand','method'])[diagnostic_cols].mean().reset_index().to_csv(OUT/'weight_dependence_diagnostics.csv',index=False)
    rows.groupby(['dataset','regime','estimand','method']).agg(MAE=('absolute_error','mean'),coverage=('coverage','mean'),half_width=('half_width','mean'),tasks=('task_id','size')).reset_index().to_csv(OUT/'regime_summary.csv',index=False)
    doc=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.7cm]{geometry}
\usepackage{booktabs}
\usepackage[T1]{fontenc}
\begin{document}
\title{Real-data KMM: target means and fixed-model risks}
\author{}\date{}\maketitle
\noindent This is an empirical stress test. Each predictor is trained separately and frozen.
Target labels are revealed only for evaluation. All errors and interval widths are in normalized units.
Tasks share stations or patients; they are not independent repetitions.
\small
'''
    doc+='\n\\clearpage\n'.join(latex_tables)
    doc+=r'''
\noindent PM2.5: 12 stations, four quarters, source 2015 and target 2016; predictor training 2013--2014.
NOAA: disjoint training, source, and target station pools in 2025; same-season single-station transfer at three distance regimes.
TNBC: eight training patients and 26 evaluation patients; three local windows per target patient.
The mean endpoint is clipped PM2.5, clipped TMAX, or clipped processed Ki67 intensity.
Ki67 and derived phenotypes are excluded from TNBC predictors.
Risk is squared prediction error on the bounded response.

\medskip\noindent Dependence correction uses calendar-lag Bartlett HAC (14 days) for time series and a Gaussian spatial covariance taper for cells.
Corrected variance is floored at the independent estimate. Weights are treated as fixed for these plug-in intervals;
weight-fitting uncertainty and target-population uncertainty are not fully included.
The reference is the observed finite target-window quantity. Coverage is descriptive and is not verification of the theorem.
\end{document}
'''
    (OUT/'paper_tables.tex').write_text(doc,encoding='utf8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--dataset',choices=['all','PM25','NOAA','TNBC'],default='all'); parser.add_argument('--limit',type=int,default=0); parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        tables(); return
    with threadpool_limits(limits=1):
        for name,loader in [('PM25',load_pm),('NOAA',load_noaa),('TNBC',load_tnbc)]:
            if args.dataset not in ['all',name]: continue
            train,features,tasks,protocol=loader()
            model=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=100,min_samples_leaf=10,max_features=1.,random_state=SEED,n_jobs=1))
            model.fit(train[features].to_numpy(),train.response.to_numpy())
            protocol.update(features=features,n_train=len(train),random_state=SEED,n_tasks=len(tasks),KMM_B=B)
            (OUT/f'{name}_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
            checkpoint=OUT/f'{name}_tasks.csv'
            result=pd.read_csv(checkpoint).to_dict('records') if checkpoint.exists() else []
            done={r['task_id'] for r in result}
            start=time.time()
            for i,(task,src,tgt,extra) in enumerate(tasks[:args.limit or None]):
                if task in done: continue
                result.extend(evaluate(name,task,src,tgt,features,model,extra,i==0))
                pd.DataFrame(result).to_csv(OUT/f'{name}_tasks.csv',index=False)
                if i%5==0: print(name,i+1,'/',len(tasks),'elapsed',round(time.time()-start),flush=True)
            tables()
            print(name,'COMPLETE',len(result),'rows',flush=True)


if __name__=='__main__': main()
