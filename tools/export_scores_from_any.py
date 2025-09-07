#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust exporter:
- Reads Parquet (single file OR glob pattern) or CSV.
- Auto-detects/renames label column to 'label' if it's named like isFraud/Class/etc.
- Transforms with provided Spark PipelineModels (RF/GBT), then writes:
    results/scores_<model>.csv   (label,score)
    results/shap_data_<model>.npz (X,y) for downstream SHAP PNGs
Usage examples:
  spark-submit tools/export_scores_from_any.py \
    --data "./data/processed_10m/valid/*.snappy.parquet" \
    --rf-pipeline models/spark_baselines_10m/rf \
    --gbt-pipeline models/spark_baselines_10m/gbt

  # or against a single file:
  spark-submit tools/export_scores_from_any.py \
    --data ./data/processed/test/part-00000-ec5259a1-2839-49b9-b156-d5f82682b423-c000.snappy.parquet \
    --rf-pipeline models/spark_baselines_10m/rf \
    --gbt-pipeline models/spark_baselines_10m/gbt
"""
import os, json, glob, shutil, argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.utils import AnalysisException
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="Parquet path (file) or glob '*.parquet', or CSV path.")
ap.add_argument("--csv", action="store_true", help="Treat --data as CSV input (otherwise Parquet).")
ap.add_argument("--csv-header", action="store_true", help="CSV has header.")
ap.add_argument("--csv-delim", default=",", help="CSV delimiter (default ,).")

ap.add_argument("--rf-pipeline", default="models/spark_baselines_10m/rf")
ap.add_argument("--gbt-pipeline", default="models/spark_baselines_10m/gbt")
ap.add_argument("--shap-n", type=int, default=10000, help="Rows sampled for SHAP NPZ (default 10k).")
args = ap.parse_args()

spark = SparkSession.builder.appName("export_scores_from_any").getOrCreate()
RES = Path("results"); RES.mkdir(exist_ok=True)

def read_input(path: str):
    p = Path(path)
    if args.csv:
        return spark.read.csv(p.as_posix(), header=args.csv_header, inferSchema=True, sep=args.csv_delim)
    # Parquet: allow glob or single file
    return spark.read.parquet(path)

def ensure_label_col(df):
    if "label" in df.columns:
        return df
    candidates = ["isFraud","is_fraud","fraud","Fraud","Class","target","y"]
    for c in candidates:
        if c in df.columns:
            return df.withColumn("label", F.col(c).cast("int"))
    raise SystemExit("[error] No 'label' column found and no known alias (isFraud/is_fraud/Class/...) present. "
                     "Add/rename your label column or tell me its name to tweak the script.")

def export_scores(pred_df, model_name: str):
    has_prob = "probability" in pred_df.columns
    has_raw  = "rawPrediction" in pred_df.columns
    if not has_prob and not has_raw:
        raise SystemExit(f"[error] {model_name}: transform produced no 'probability'/'rawPrediction' columns.")
    if "label" not in pred_df.columns:
        raise SystemExit(f"[error] {model_name}: no 'label' column post-transform.")

    if has_prob:
        sel = pred_df.select(F.col("label").cast("int").alias("label"),
                             vector_to_array("probability")[1].alias("score"))
    else:
        sel = pred_df.select(F.col("label").cast("int").alias("label"),
                             (1.0 / (1.0 + F.exp(-vector_to_array("rawPrediction")[0]))).alias("score"))
    tmp_dir = f"results/_scores_{model_name}_tmp"
    out_csv = RES / f"scores_{model_name}.csv"
    sel.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_dir)
    part = glob.glob(os.path.join(tmp_dir, "part-*.csv"))[0]
    shutil.move(part, out_csv.as_posix())
    shutil.rmtree(tmp_dir)
    print(f"[ok] wrote {out_csv}")

def export_shap_npz(pred_df, model_name: str, n_rows: int):
    if "features" not in pred_df.columns:
        print(f"[warn] {model_name}: no 'features' after transform; skipping SHAP NPZ")
        return
    sub = pred_df.select(vector_to_array("features").alias("x"),
                         F.col("label").cast("int").alias("y")).limit(n_rows)
    pdf = sub.toPandas()
    import numpy as np
    if len(pdf) == 0:
        print(f"[warn] {model_name}: empty sample for SHAP NPZ"); return
    X = np.stack(pdf["x"].values); y = pdf["y"].to_numpy()
    out_npz = RES / f"shap_data_{model_name}.npz"
    np.savez_compressed(out_npz, X=X, y=y)
    print(f"[ok] wrote {out_npz} with X.shape={X.shape}, y.shape={y.shape}")

# ---- Read input ----
try:
    df_in = read_input(args.data)
except Exception as e:
    raise SystemExit(f"[error] reading {args.data}: {e}")

# Ensure a label col exists for downstream KPI alignment (also helps some pipelines)
try:
    df_in = ensure_label_col(df_in)
except SystemExit as e:
    print(str(e))
    spark.stop(); raise

# ---- Load and transform with each pipeline if it exists ----
pipelines = {
    "rf":  args.rf_pipeline,
    "gbt": args.gbt_pipeline,
}
for model_code, model_path in pipelines.items():
    mp = Path(model_path)
    if not mp.exists():
        print(f"[skip] {model_code}: {mp} not found"); continue

    print(f"[..] loading PipelineModel {model_code}: {mp}")
    pipe = PipelineModel.load(mp.as_posix())

    print(f"[..] transforming input")
    try:
        pred = pipe.transform(df_in)
    except AnalysisException as e:
        spark.stop()
        raise SystemExit(f"[error] transform failed for {model_code}: {e.desc}\n"
                         f"Hint: input columns must match those used in training.")

    export_scores(pred, model_code)
    export_shap_npz(pred, model_code, n_rows=args.shap_n)

spark.stop()
