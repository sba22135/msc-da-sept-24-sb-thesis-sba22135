# FraudLab: Practical Pipeline for Explainable Deep Learning Fraud Detection (Spark + PyTorch)

This project gives you **step-by-step, runnable code** for your thesis:
- Synthetic PaySim-like data generation (configurable size).
- PySpark preprocessing with train/valid/test split and class-imbalance handling.
- Baseline ML models in Spark (LogReg, RandomForest, GBT) with metrics.
- Sequence building for deep learning and **PyTorch LSTM** training.
- XAI: SHAP (tree & neural) + LIME for local explanations (on a sample).
- Streamlit dashboard to explore metrics, confusion matrices, and SHAP outputs.

> ⚠️ Scale carefully. Start small (e.g., 100k rows), then scale up as your VM allows.
> GPU is optional; CPU will work but be slower. Avoid crashing Firefox by running scripts via terminal (not in-browser).

## 0) Environment & Quick Start

**Prereqs installed on your Ubuntu VM:** Spark 3.x, Hadoop/HDFS, Python 3.10+, pip, (optional) CUDA/GPU.

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements (skip packages you already have)
pip install -r requirements.txt
```

Optional: if you want to use HDFS:
```bash
# Make an HDFS directory for the project
hdfs dfs -mkdir -p /user/$USER/fraudlab

# (You can copy data later with: hdfs dfs -put <localfile> /user/$USER/fraudlab/ )
```

## 1) Generate Synthetic PaySim-like Data

```bash
python spark_jobs/01_generate_paysim_synthetic.py --out data/raw/paysim_synth.csv --n_rows 100000
# Scale up later, e.g.: --n_rows 10000000
```

Optionally copy to HDFS:
```bash
hdfs dfs -put -f data/raw/paysim_synth.csv /user/$USER/fraudlab/paysim_synth.csv
```

## 2) Preprocess in Spark

```bash
python spark_jobs/02_preprocess_spark.py   --input data/raw/paysim_synth.csv   --out_dir data/processed   --use_hdfs 0
```

This writes train/valid/test Parquet files with encoded features and a `weightCol` to mitigate class imbalance.

## 3) Train Baseline ML Models (Spark)

```bash
python spark_jobs/03_train_baselines_spark.py   --data_dir data/processed   --models_dir models/spark_baselines   --metrics_path results/spark_baseline_metrics.json
```

## 4) Evaluate & Export Confusion Matrices

```bash
python spark_jobs/04_evaluate_spark.py   --data_dir data/processed   --models_dir models/spark_baselines   --results_dir results
```

## 5) Build Sequences for Deep Learning (PySpark)

```bash
python deep_learning/10_build_sequences_pyspark.py   --input data/raw/paysim_synth.csv   --out data/processed/sequences.parquet   --seq_len 20
```

## 6) Train LSTM (PyTorch)

```bash
python deep_learning/11_train_lstm_pytorch.py   --seq_path data/processed/sequences.parquet   --out_dir models/pytorch_lstm   --epochs 3   --batch_size 512
```

## 7) Explainability (SHAP & LIME)

```bash
# SHAP for Spark tree model(s) on a sample
python deep_learning/12_explain_shap_tree.py   --models_dir models/spark_baselines   --data_dir data/processed   --out_dir results/xai_tree

# SHAP (Kernel) + LIME for LSTM on a small sample
python deep_learning/12_explain_shap_pytorch.py   --model_dir models/pytorch_lstm   --seq_path data/processed/sequences.parquet   --out_dir results/xai_lstm   --sample_size 500
```

## 8) Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The app reads `results/` & `models/` artifacts and lets you switch models, view metrics, confusion matrices and SHAP summaries.

---

### Tips
- If RAM is tight, lower `--n_rows`, `--batch_size`, or sample for XAI.
- For Spark jobs, prefer running from a terminal (not Jupyter).
- Keep raw CSV locally; store large processed files in HDFS if you have space.
- Save screenshots/figures from `results/` for your thesis Chapters 4–5.
