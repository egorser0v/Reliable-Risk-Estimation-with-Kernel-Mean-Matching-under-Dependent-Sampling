#!/usr/bin/env bash
set -euo pipefail

OUT="audit_context.txt"
rm -f "$OUT"

echo "=== GATHERING PROJECT CONTEXT FOR AUDIT ==="

append_file() {
    local fpath="$1"
    if [ -f "$fpath" ]; then
        echo "" >> "$OUT"
        echo "================================================================================" >> "$OUT"
        echo "FILE: $fpath" >> "$OUT"
        echo "================================================================================" >> "$OUT"
        cat "$fpath" >> "$OUT"
        echo "" >> "$OUT"
    fi
}

append_head() {
    local fpath="$1"
    local lines="${2:-10}"
    if [ -f "$fpath" ]; then
        echo "" >> "$OUT"
        echo "================================================================================" >> "$OUT"
        echo "HEAD OF CSV ($lines lines): $fpath" >> "$OUT"
        echo "================================================================================" >> "$OUT"
        head -n "$lines" "$fpath" >> "$OUT"
        echo "" >> "$OUT"
    fi
}

# 1. Документация и окружение
append_file "README.md"

# 2. Основные скрипты экспериментов
append_file "experiments/kmm_three_domain_benchmark.py"
append_file "experiments/verify_kmm_three_domain.py"

# 3. Отчеты и сводные таблицы из results
append_file "results/kmm_three_domain/report.md"
append_file "results/kmm_three_domain/paper_tables.tex"
append_file "results/kmm_three_domain/mean_summary.csv"
append_file "results/kmm_three_domain/risk_summary.csv"
append_file "results/kmm_three_domain/regime_summary.csv"

# 4. JSON-протоколы (там настройки сплитов, таргетов и гиперпараметров)
append_file "results/kmm_three_domain/NOAA_protocol.json"
append_file "results/kmm_three_domain/PM25_protocol.json"
append_file "results/kmm_three_domain/TNBC_protocol.json"

# 5. Первые строки датасетов, чтобы увидеть структуру колонок (X и Y)
append_head "data/noaa/ghcn2025_benchmark.csv" 10
append_head "data/pm25/beijing_pm25_daily_prepared.csv" 10
append_file "data/tnbc/provenance.json"

# 6. Первые строки тасковых csv результатов (понять, что за метрики и таски)
append_head "results/kmm_three_domain/NOAA_tasks.csv" 5
append_head "results/kmm_three_domain/PM25_tasks.csv" 5
append_head "results/kmm_three_domain/TNBC_tasks.csv" 5

echo "Done! Context collected in $OUT"
ls -lh "$OUT"
