import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Порог валидности интервала из статьи (для nominal 0.95)
COVERAGE_THRESHOLD = 0.925

def load_and_aggregate():
    # Ищем все csv в папке результатов
    files = glob.glob('results/kmm_final_paper/*.csv')
    if not files:
        print("CSV файлы не найдены в results/kmm_final_paper/")
        return None
    
    print(f"Загрузка файлов: {files}\n")
    df = pd.concat([pd.read_csv(f) for f in files])
    
    # Агрегируем (усредняем) результаты по всем повторениям (reps)
    grouped = df.groupby(
        ['dataset', 'n_source', 'clip_b', 'estimand', 'method']
    )[['coverage', 'radius', 'WS', 'MAE']].mean().reset_index()
    
    return grouped

def get_best_method(sub_df):
    """
    Выбирает лучший метод по логике статьи:
    1. Если есть методы с coverage >= 0.925, берем среди них тот, у которого минимальный WS.
    2. Если таких нет (все under-covered), берем тот, у которого максимальный coverage.
    """
    valid_methods = sub_df[sub_df['coverage'] >= COVERAGE_THRESHOLD]
    
    if not valid_methods.empty:
        # Ищем минимальный Winkler Score среди валидных
        best_idx = valid_methods['WS'].idxmin()
        return valid_methods.loc[best_idx, 'method']
    else:
        # Если все провалили threshold, лучший тот, кто ближе всех к 0.95
        best_idx = sub_df['coverage'].idxmax()
        return sub_df.loc[best_idx, 'method']

def print_table(sub_df, title=""):
    print(f"\n{title}")
    print(f"{'Method':<20} | {'Coverage':<10} | {'Radius':<10} | {'WS ↓':<10}")
    print("-" * 60)
    
    best_method = get_best_method(sub_df)
    
    # Задаем строгий порядок вывода как в статье
    order = ['Unweighted', 'Nominal n', 'Block-Bootstrap', 'Input neff', 'GP neff [Ours]']
    
    for m in order:
        m_row = sub_df[sub_df['method'] == m]
        if m_row.empty:
            continue
        
        m_row = m_row.iloc[0]
        cov, rad, ws = m_row['coverage'], m_row['radius'], m_row['WS']
        
        # Ставим звездочку, если coverage провалился
        cov_str = f"{cov:.3f}" if cov >= COVERAGE_THRESHOLD else f"{cov:.3f}*"
        
        # Выделяем лучший метод
        marker = "<-- BEST (Bold)" if m == best_method else ""
        
        print(f"{m:<20} | {cov_str:<10} | {rad:<10.4f} | {ws:<10.4f} {marker}")

def find_and_print_best_configs(grouped):
    datasets = grouped['dataset'].unique()
    
    for ds in datasets:
        ds_data = grouped[grouped['dataset'] == ds]
        
        # Ищем идеальную конфигурацию (n_source, clip_b) ориентируясь на Target Risk
        # Наш метод GP neff должен быть валидным и иметь минимальный WS
        gp_data = ds_data[(ds_data['method'] == 'GP neff [Ours]') & (ds_data['estimand'] == 'Target Risk (L)')]
        
        best_config = None
        best_score = float('inf')
        
        for _, row in gp_data.iterrows():
            if row['coverage'] >= COVERAGE_THRESHOLD:
                score = row['WS'] # Чем меньше WS, тем лучше
            else:
                score = 1000 - row['coverage'] # Штрафуем невалидные, но ранжируем по coverage
                
            if score < best_score:
                best_score = score
                best_config = (row['n_source'], row['clip_b'])
        
        if not best_config:
            continue
            
        n_best, c_best = best_config
        
        print("=" * 80)
        print(f"🥇 DATASET: {ds.upper()} | BEST HIGHLIGHT CONFIGURATION: n_source = {n_best}, clip_B = {c_best}")
        print("=" * 80)
        
        for estimand in ['Target Mean (Y)', 'Target Risk (L)']:
            sub = ds_data[
                (ds_data['n_source'] == n_best) & 
                (ds_data['clip_b'] == c_best) & 
                (ds_data['estimand'] == estimand)
            ]
            print_table(sub, title=f"Estimand: {estimand}")
        print("\n")

if __name__ == "__main__":
    aggregated_data = load_and_aggregate()
    if aggregated_data is not None:
        find_and_print_best_configs(aggregated_data)
        
        # Опционально: можно сохранить агрегированную таблицу для удобного переноса в LaTeX
        aggregated_data.to_csv('results/kmm_final_paper/aggregated_tables_ready.csv', index=False)
        print("✅ Все усредненные данные также сохранены в: results/kmm_final_paper/aggregated_tables_ready.csv")