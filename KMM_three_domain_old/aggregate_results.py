import pandas as pd
import numpy as np
import glob

# Порог валидности интервала, как указано в статье (0.95 - 1.96 * sqrt(...))
COVERAGE_THRESHOLD = 0.925

def load_and_aggregate():
    files = glob.glob('results/kmm_final_paper/*.csv')
    if not files:
        print("CSV файлы не найдены в results/kmm_final_paper/")
        return None
    
    # Читаем все файлы и склеиваем в один DataFrame
    df = pd.concat([pd.read_csv(f) for f in files])
    
    # Усредняем (mean) по всем повторениям (reps)
    grouped = df.groupby(
        ['dataset', 'n_source', 'clip_b', 'estimand', 'method']
    )[['coverage', 'radius', 'WS', 'MAE']].mean().reset_index()
    
    return grouped

def get_best_method(sub_df):
    """
    Алгоритм выбора лучшего метода по правилам:
    1. Приоритет тем, у кого coverage >= 0.925. Среди них выбирается минимальный radius.
    2. Если никого с coverage >= 0.925 нет, выбирается максимальный coverage.
    """
    valid_methods = sub_df[sub_df['coverage'] >= COVERAGE_THRESHOLD]
    
    if not valid_methods.empty:
        # Если есть валидные методы -> берем тот, у кого НАИМЕНЬШИЙ РАДИУС
        best_idx = valid_methods['radius'].idxmin()
        return valid_methods.loc[best_idx, 'method']
    else:
        # Если валидных нет -> берем тот, у кого НАИБОЛЬШИЙ COVERAGE
        best_idx = sub_df['coverage'].idxmax()
        return sub_df.loc[best_idx, 'method']

def print_all_tables(grouped):
    datasets = sorted(grouped['dataset'].unique())
    
    # Фиксированный порядок методов для красивого вывода как в статье
    method_order = [
        'Unweighted', 'Nominal n', 'Block-Bootstrap', 
        'Input neff', 'GP neff [Ours]', 
        'Strict Bound (Theorem 1)', 'Oracle IW', 'Oracle i.i.d.'
    ]
    
    for ds in datasets:
        ds_data = grouped[grouped['dataset'] == ds]
        n_sources = sorted(ds_data['n_source'].unique())
        clip_bs = sorted(ds_data['clip_b'].unique())
        
        for n_src in n_sources:
            for cb in clip_bs:
                print(f"\n{'='*80}")
                print(f"DATASET: {ds.upper()} | n_source: {n_src} | clip_B: {cb}")
                print(f"{'='*80}")
                
                for estimand in ['Target Mean (Y)', 'Target Risk (L)']:
                    sub = ds_data[
                        (ds_data['n_source'] == n_src) & 
                        (ds_data['clip_b'] == cb) & 
                        (ds_data['estimand'] == estimand)
                    ]
                    
                    if sub.empty:
                        continue
                    
                    print(f"\nEstimand: {estimand}")
                    print(f"{'Method':<20} | {'Coverage':<10} | {'Radius':<10} | {'WS ↓':<10}")
                    print("-" * 65)
                    
                    best_method = get_best_method(sub)
                    
                    # Чтобы методы выводились в правильном порядке, идем по method_order
                    present_methods = sub['method'].unique()
                    
                    for m in method_order:
                        if m not in present_methods:
                            continue
                            
                        m_row = sub[sub['method'] == m].iloc[0]
                        cov = m_row['coverage']
                        rad = m_row['radius']
                        ws = m_row['WS']
                        
                        # Помечаем звездочкой провалившийся coverage
                        cov_str = f"{cov:.3f}" if cov >= COVERAGE_THRESHOLD else f"{cov:.3f}*"
                        
                        # Помечаем лучший метод
                        marker = "<-- BEST" if m == best_method else ""
                        
                        print(f"{m:<20} | {cov_str:<10} | {rad:<10.4f} | {ws:<10.4f} {marker}")
                print("\n")

if __name__ == "__main__":
    aggregated_data = load_and_aggregate()
    if aggregated_data is not None:
        print_all_tables(aggregated_data)
        
        # Сохраним CSV для истории
        aggregated_data.to_csv('results/kmm_final_paper/aggregated_all_tables.csv', index=False)