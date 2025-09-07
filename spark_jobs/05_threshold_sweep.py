#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Threshold sweep for Spark classifiers (LR / RF / GBT).
- Auto-resolves model path (pipeline or bare model).
- Extracts P(class=1) robustly from the 'probability' vector.
- Computes precision/recall/F1 across thresholds + PR-AUC.
- Writes a compact JSON for reporting.

Usage example:
  python spark_jobs/05_threshold_sweep.py \
    --data_dir data/processed_10m \
    --models_dir models/spark_baselines_10m \
    --model logreg \
    --out results/threshold_sweep_logreg.json \
    --master local[2]
"""

import os, json, argparse
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    LogisticRegressionModel,
    RandomForestClassificationModel,
    GBTClassificationModel,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator

CANDIDATE_NAMES = {
    "logreg": ["logreg", "lr", "baseline_logreg"],
    "rf":     ["rf", "random_forest", "randomForest"],
    "gbt":    ["gbt", "gbtc", "gradient_boosted_trees"],
}

def resolve_model_path(models_dir: str, name: str) -> str:
    """Find a likely subfolder for the requested model name."""
    for n in CANDIDATE_NAMES[name] + [name]:
        p = os.path.join(models_dir, n)
        if os.path.exists(p):
            return p
    available = [d for d in os.listdir(models_dir)
                 if os.path.isdir(os.path.join(models_dir, d))]
    raise FileNotFoundError(
        f"Model folder for '{name}' not found under {models_dir}. "
        f"Looked for {CANDIDATE_NAMES[name] + [name]}. Available: {available}"
    )

def load_any_model(path: str, kind: str):
    """Load either a bare model or a PipelineModel from disk."""
    # Try bare model first
    try:
        if kind == "logreg":
            return LogisticRegressionModel.load(path)
        if kind == "rf":
            return RandomForestClassificationModel.load(path)
        if kind == "gbt":
            return GBTClassificationModel.load(path)
    except Exception:
        pass
    # Fallback to pipeline
    return PipelineModel.load(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--model", choices=["logreg","rf","gbt"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--master", default="local[2]")
    args = ap.parse_args()

    spark = (SparkSession.builder
             .master(args.master)
             .appName("threshold_sweep")
             .getOrCreate())

    # Load test set
    test = spark.read.parquet(os.path.join(args.data_dir, "test"))

    # Load model (pipeline or bare)
    model_path = resolve_model_path(args.models_dir, args.model)
    mdl = load_any_model(model_path, args.model)

    # Predict and extract P(class=1) from VectorUDT 'probability'
    getprob = F.udf(lambda v: float(v[1]) if v is not None else 0.0, DoubleType())
    pred_full = mdl.transform(test)

    if "probability" not in pred_full.columns:
        raise RuntimeError("Predictions missing 'probability' column; check saved model/pipeline.")

    pred = pred_full.select(
        F.col("label").cast("int").alias("label"),
        getprob(F.col("probability")).alias("p1")
    )

    # Sweep thresholds
    thresholds = [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    rows = []
    for t in thresholds:
        y = pred.select("label", (F.col("p1") >= F.lit(t)).cast("int").alias("yhat"))
        tp = y.filter("label=1 AND yhat=1").count()
        fp = y.filter("label=0 AND yhat=1").count()
        fn = y.filter("label=1 AND yhat=0").count()
        tn = y.filter("label=0 AND yhat=0").count()

        prec = (tp/(tp+fp)) if (tp+fp) > 0 else 0.0
        rec  = (tp/(tp+fn)) if (tp+fn) > 0 else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0

        rows.append({
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": [[tn, fp], [fn, tp]]
        })

    # PR-AUC reference (use probability vector)
    pr_auc = BinaryClassificationEvaluator(
        metricName="areaUnderPR", labelCol="label", rawPredictionCol="probability"
    ).evaluate(pred_full)

    out = {
        "model": args.model,
        "model_path": model_path,
        "pr_auc": pr_auc,
        "sweep": rows
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    spark.stop()

if __name__ == "__main__":
    main()
