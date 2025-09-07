# spark_jobs/04_evaluate_spark.py  (AUC disabled for stability)
import argparse, os, json
from pyspark.ml.classification import (
    LogisticRegressionModel,
    RandomForestClassificationModel,
    GBTClassificationModel,
)
from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def get_spark(app_name="FraudLab-Evaluate-Baselines", master=None):
    b = SparkSession.builder.appName(app_name)
    if master:
        b = b.master(master)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def eval_model(model, test_df):
    """Compute accuracy/precision/recall/F1 + confusion matrix (no AUC)."""
    getprob = F.udf(lambda v: float(v[1]), "double")

    pred = model.transform(test_df).withColumn("prob1", getprob(F.col("probability")))
    pred = pred.select("probability", "prob1", F.col("label").cast("int").alias("label"))
    pred = pred.persist(StorageLevel.MEMORY_AND_DISK)

    # threshold @ 0.5
    yhat = (F.col("prob1") >= F.lit(0.5)).cast("int")
    label = F.col("label")

    TP = F.sum(F.when((yhat == 1) & (label == 1), 1).otherwise(0)).alias("TP")
    FP = F.sum(F.when((yhat == 1) & (label == 0), 1).otherwise(0)).alias("FP")
    TN = F.sum(F.when((yhat == 0) & (label == 0), 1).otherwise(0)).alias("TN")
    FN = F.sum(F.when((yhat == 0) & (label == 1), 1).otherwise(0)).alias("FN")

    counts = pred.agg(TP, FP, TN, FN).collect()[0].asDict()
    pred.unpersist()

    tp, fp, tn, fn = int(counts["TP"]), int(counts["FP"]), int(counts["TN"]), int(counts["FN"])
    total = tp + fp + tn + fn

    def safe(a, b): return float(a) / b if b else 0.0
    precision = safe(tp, tp + fp)
    recall    = safe(tp, tp + fn)
    f1        = safe(2 * precision * recall, precision + recall)

    metrics = {
        "accuracy":  safe(tp + tn, total),
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "auc":       None  # intentionally skipped for stability
    }

    cm = [[tn, fp], [fn, tp]]

# ===== Extra metrics: PR-AUC & ROC-AUC =====
    try:
        pr_eval = BinaryClassificationEvaluator(
            metricName="areaUnderPR", labelCol="label", rawPredictionCol="probability"
        )
        metrics["auc_pr"] = float(pr_eval.evaluate(pred))

        roc_eval = BinaryClassificationEvaluator(
            metricName="areaUnderROC", labelCol="label", rawPredictionCol="probability"
        )
        metrics["auc_roc"] = float(roc_eval.evaluate(pred))
    except Exception as e:
        metrics["auc_error"] = f"{type(e).__name__}: {e}"

# ===== Feature importances / coefficients =====
    try:
    # handle pipeline vs direct model
        last = model.stages[-1] if hasattr(model, "stages") else model
        clsname = last.__class__.__name__

    # Logistic Regression coefficients
        if clsname == "LogisticRegressionModel":
            coefs = last.coefficients.toArray().tolist()
            metrics["lr_coeff_top"] = sorted(
                [(f"f{i}", float(w)) for i, w in enumerate(coefs)],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]

    # Tree-based models
        if clsname in ("RandomForestClassificationModel", "GBTClassificationModel"):
            imps = last.featureImportances
            metrics["tree_importance_top"] = sorted(
                [(f"f{i}", float(imps[i])) for i in range(len(imps))],
                key=lambda x: x[1],
                reverse=True
            )[:20]
    except Exception as e:
        metrics["explain_error"] = f"{type(e).__name__}: {e}"

    return metrics, cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--metrics_path", required=True)
    ap.add_argument("--master", default=None)
    args = ap.parse_args()

    spark = get_spark(master=args.master)

    test_path = "file:///" + os.path.join(os.getcwd(), args.data_dir, "test")
    test = spark.read.parquet(test_path).repartition(4)

    results = {}
    model_map = {
        "logreg": ("lr",  LogisticRegressionModel),
        "rf":     ("rf",  RandomForestClassificationModel),
        "gbt":    ("gbt", GBTClassificationModel),
    }

    for pretty, (folder, Loader) in model_map.items():
        local_dir = os.path.join(os.getcwd(), args.models_dir, folder)
        if not os.path.isdir(local_dir):
            print(f"  Skipping {pretty}: {local_dir} not found.")
            continue
        model_uri = "file:///" + local_dir
        model = Loader.load(model_uri)
        m, cm = eval_model(model, test)
        results[pretty] = {"metrics": m, "confusion_matrix": cm}

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f" Saved test metrics to: {args.metrics_path}")

if __name__ == "__main__":
    main()

