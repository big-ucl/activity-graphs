#!/usr/bin/env bash
# Run with nohup ./overnight.sh > /dev/null 2>&1 &

cd "$(dirname "$0")"
log="reports/logs/overnight-$(date +%F).log"

for data in cmap geneva; do
    echo "=== $data smoke $(date) ===" >> "$log"
    uv run activity-graphs data=$data train.experiment=comparison \
        train.fast_dev_run=true train.wandb=false >> "$log" 2>&1 || exit 1
done

for data in cmap geneva; do
    echo "=== $data start $(date) ===" >> "$log"
    uv run activity-graphs data=$data train.experiment=comparison >> "$log" 2>&1
    echo "=== $data exit $? $(date) ===" >> "$log"

    echo "=== $data report $(date) ===" >> "$log"
    uv run report-results data=$data >> "$log" 2>&1
done