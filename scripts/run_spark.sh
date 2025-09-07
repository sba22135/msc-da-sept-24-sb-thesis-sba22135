#!/usr/bin/env bash
set -e
DATA_DIR="data/processed_10m"
MODELS_DIR="models/spark_baselines_10m"
METRICS_PATH="results/spark_baseline_metrics_10m.json"

python spark_jobs/03_train_baselines_spark.py \
  --data_dir "$DATA_DIR" \
  --models_dir "$MODELS_DIR" \
  --metrics_path "$METRICS_PATH" \
  --master local[*] \
  --train_frac 0.05
