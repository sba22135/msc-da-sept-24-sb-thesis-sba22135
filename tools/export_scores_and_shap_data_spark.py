#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export scores_<model>.csv (label, score) and shap_data_<model>.npz (X,y) by
loading a saved Spark PipelineModel and transforming a dataset (parquet/csv).

Usage:
  spark-submit tools/export_scores_and_shap_data_spark.py \
    --data ./data/processed_10m/valid
"""
import os, json, glob, shutil, argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.utils import AnalysisException
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

# Your discovered models:
MODEL_DIRS = {
    "rf":  "models/spark_baselines_10m/rf",
    "gbt": "models/spark_baselines_10m/gbt",
    # "logreg": "models/spark_baselines_10m/logreg",  # add if you have it
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True,
                help="Path to dataset directory/file (parquet or csv) compatible with pipeline schema.")
ap.add_argument("--csv-header", action="store_true",
                help="Set if --data is CSV and the file has a header row.")
ap.add_argument("--csv-delim", default=",",
                help="CSV delimiter (default ,).")
ap.add_argument("--shap-n", type=int, default=10000,
                help="Rows to sample for SHAP NPZ (default 10000).")
args = ap.parse_args()

spark = SparkSession.builder.appName("export_scores_and_shap_data").getOrCreate()
RES = Path("results"); RES.mkdir(exist_ok=True)

def read_any(path: str):
    p = Path(path)
    if not p.exists():
        # also allow directory path (e.g., ./data/processed_10m/valid)
        if Path(path).is_dir():
            # Spark will still read the folder if it exists; fall through
            pass
    # Try parquet first (works on folder or single file)
    try:
        return spark.read.parquet(path)
    except Exception:
        # fallback to CSV
        if Path(path).is_dir():
            raise SystemExit(f"[error] expected parquet folder or file at: {path}")
        return spark.read.csv(path, header=args.csv_header, inferSchema=True, sep=args.csv_delim)

def export_scores(pred_df, model_name: str, has_prob: bool):
    if "label" not in pred_df.columns:
        raise SystemExit(f"[error] transformed dataframe has no 'label' column; check input schema/pipeline.")
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
        print(f"[warn] {model_name}: no 'features' column after transform; skipping SHAP NPZ")
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

def maybe_dump_lr_params(pipeline_model):
    try:
        from pyspark.ml.classification import LogisticRegressionModel
    except Exception:
        LogisticRegressionModel = None
    if LogisticRegressionModel is None: return
    for st in reversed(pipeline_model.stages):
        if isinstance(st, LogisticRegressionModel):
            coef = list(st.coefficients.toArray())
            intercept = float(st.intercept)
            Path("results/spark_logreg_params.json").write_text(
                json.dumps({"coefficients": coef, "intercept": intercept}, indent=2))
            print("[ok] wrote results/spark_logreg_params.json")
            return

df_in = read_any(args.data)

for code, model_path in MODEL_DIRS.items():
    mp = Path(model_path)
    if not mp.exists():
        print(f"[skip] {code}: {mp} not found"); continue
    print(f"[..] loading PipelineModel for {code}: {mp}")
    pipe = PipelineModel.load(mp.as_posix())
    print(f"[..] transforming input data")
    try:
        pred = pipe.transform(df_in)
    except AnalysisException as e:
        raise SystemExit(f"[error] transform failed for {code}: {e.desc}\n"
                         f"Hint: ensure --data has same input columns as during training.")
    has_prob = ("probability" in pred.columns)
    if not has_prob and "rawPrediction" not in pred.columns:
        print(f"[warn] {code}: no probability/rawPrediction columns found; skipping score export.")
    else:
        export_scores(pred, code, has_prob=has_prob)
    export_shap_npz(pred, code, n_rows=args.shap_n)
    if code == "logreg":
        maybe_dump_lr_params(pipe)

spark.stop()
