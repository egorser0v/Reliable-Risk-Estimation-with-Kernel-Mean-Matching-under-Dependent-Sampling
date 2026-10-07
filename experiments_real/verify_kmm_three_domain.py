"""Check numerical feasibility, shared kernel, and absence of target-label leakage."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from kmm_three_domain_benchmark import weights, evaluate, correction_matrix, OUT


class FixedModel:
    def predict(self,x): return np.full(len(x),.4)


with threadpool_limits(limits=1):
    rng=np.random.default_rng(22)
    xs=rng.normal(size=(25,3)); xt=rng.normal(.2,size=(23,3))
    ws,mmd,info=weights(xs,xt,True)
    assert abs(ws['KMM'].sum()-25)<1e-6
    assert info['solver_objective_relative_difference']<1e-4
    assert mmd(ws['KMM'])<=mmd(np.ones(25))+1e-7
    src=pd.DataFrame(xs,columns=['a','b','c']); tgt=pd.DataFrame(xt,columns=['a','b','c'])
    for df in [src,tgt]:
        df['response']=rng.uniform(size=len(df)); df['domain']='test'
        df['date']=pd.date_range('2020-01-01',periods=len(df))
    a=pd.DataFrame(evaluate('PM25','test',src,tgt,['a','b','c'],FixedModel(),{},False))
    altered=tgt.copy(); altered['response']=1-altered.response
    b=pd.DataFrame(evaluate('PM25','test',src,altered,['a','b','c'],FixedModel(),{},False))
    for col in ['estimate','half_width','max_weight','mmd2_after','adjusted_ess']:
        assert np.allclose(a[col],b[col]),col
    C,_=correction_matrix(src,False)
    assert np.linalg.eigvalsh(C).min()>-1e-8
    # Identical covariates must have negligible discrepancy for unit weights.
    _,same,_=weights(xs,xs)
    assert same(np.ones(len(xs)))<1e-10
    print('PASS: constrained QP, independent solver, MMD, HAC PSD, target-label isolation')

if list(OUT.glob('*_tasks.csv')):
    for p in OUT.glob('*_tasks.csv'):
        d=pd.read_csv(p)
        assert not d[['estimate','true_target','half_width','absolute_error']].isna().any().any()
        k=d[d.method=='KMM'].set_index(['task_id','estimand'])
        adj=d[d.method=='KMM + dependence'].set_index(['task_id','estimand'])
        assert np.allclose(k.estimate,adj.estimate)
        assert (adj.half_width>=k.half_width-1e-12).all()
        assert (k.mmd2_after<=k.mmd2_before+1e-6).all()
        assert not d.duplicated(['task_id','estimand','method']).any()
        print(p.name, d.task_id.nunique(), 'tasks: PASS')
