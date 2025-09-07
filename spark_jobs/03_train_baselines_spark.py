import argparse, os, json, traceback
from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import functions as F


def get_spark(app_name="FraudLab-Train-LR-Robust", master=None):
    b = (SparkSession.builder.appName(app_name)
         .config("spark.sql.shuffle.partitions", "2")  # keep shuffles tiny
         .config("spark.default.parallelism", "2")
         .config("spark.hadoop.fs.defaultFS", "file:///"))
    if master:
        b = b.master(master)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def metrics_from_preds(df_pred):
    getprob = F.udf(lambda v: float(v[1]), "double")
    pred = (
        df_pred.withColumn("prob1", getprob(F.col("probability")))
               .select(F.col("label").cast("int").alias("label"),
                       (F.col("prob1") >= F.lit(0.5)).cast("int").alias("yhat"))
    )
    TP = F.sum(F.when((F.col("yhat")==1) & (F.col("label")==1), 1).otherwise(0)).alias("TP")
    FP = F.sum(F.when((F.col("yhat")==1) & (F.col("label")==0), 1).otherwise(0)).alias("FP")
    TN = F.sum(F.when((F.col("yhat")==0) & (F.col("label")==0), 1).otherwise(0)).alias("TN")
    FN = F.sum(F.when((F.col("yhat")==0) & (F.col("label")==1), 1).otherwise(0)).alias("FN")
    c = pred.agg(TP, FP, TN, FN).collect()[0].asDict()
    tp, fp, tn, fn = int(c["TP"]), int(c["FP"]), int(c["TN"]), int(c["FN"])
    tot = tp+fp+tn+fn
    safe = lambda a,b: float(a)/b if b else 0.0
    precision = safe(tp, tp+fp); recall = safe(tp, tp+fn)
    f1 = safe(2*precision*recall, precision+recall)
    mets = {"accuracy": safe(tp+tn, tot), "precision": precision, "recall": recall, "f1": f1, "auc": None}
    cm = [[tn, fp], [fn, tp]]
    return mets, cm

def try_fit(train_df, valid_df, use_weight=True, reg=0.1, max_iter=20):
    kwargs = dict(featuresCol="features", labelCol="label", maxIter=max_iter, tol=1e-4,
                  regParam=reg, elasticNetParam=0.0, fitIntercept=True)
    if use_weight and "weight" in train_df.columns:
        kwargs["weightCol"] = "weight"
    lr = LogisticRegression(**kwargs)
    model = lr.fit(train_df)
    mets, cm = metrics_from_preds(model.transform(valid_df))
    return model, mets, cm, kwargs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--metrics_path", required=True)
    ap.add_argument("--master", default=None)
    ap.add_argument("--train_frac", type=float, default=0.05)
    args = ap.parse_args()

    spark = get_spark(master=args.master)
    base = os.path.join(os.getcwd(), args.data_dir)

    train0 = (spark.read.parquet("file:///" + os.path.join(base, "train"))
                    .repartition(2)
                    .persist(StorageLevel.DISK_ONLY))

    pos = train0.filter("label = 1").count()
    neg = train0.filter("label = 0").count()
    pos_weight = float(neg) / float(max(pos, 1))
    train0 = train0.withColumn(
     "weight",
      F.when(F.col("label") == 1, F.lit(pos_weight)).otherwise(F.lit(1.0))
    )

    valid  = (spark.read.parquet("file:///" + os.path.join(base, "valid"))
                    .repartition(1)
                    .persist(StorageLevel.DISK_ONLY))

    # Backoff schedule: (fraction, use_weight?)
    f0 = max(0.0005, min(1.0, args.train_frac))
    schedule = [
        (f0, True),
        (min(0.02, f0/2), True),
        (min(0.02, f0/2), False),
        (0.01, False),
        (0.005, False),
    ]

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)

    last_error = None
    for frac, use_weight in schedule:
        try:
            train = train0.sample(False, frac, seed=42) if frac < 1.0 else train0
            # Small sanity action so Spark actually materializes lineage early
            _ = train.limit(1).count()

            print(f"🔧 Trying LR fit: frac={frac}, use_weight={use_weight}")
            model, mets, cm, used = try_fit(train, valid, use_weight=use_weight, reg=0.1, max_iter=20)

            out_dir = "file:///" + os.path.join(os.path.abspath(args.models_dir), "lr")
            model.write().overwrite().save(out_dir)

            results = {"logreg": {"metrics": mets, "confusion_matrix": cm,
                                  "train_frac": frac, "used_weight": use_weight}}
            with open(args.metrics_path, "w") as f:
                json.dump(results, f, indent=2)
            print(" Trained & saved. Metrics written to:", args.metrics_path)
            train0.unpersist(); valid.unpersist()
            return
        except Exception as e:
            last_error = e
            print(f"  Fit failed for frac={frac}, use_weight={use_weight}. Will back off.\n{e}\n{traceback.format_exc()}")

    # If we reach here, all attempts failed:
    train0.unpersist(); valid.unpersist()
    raise RuntimeError(f"All LR fit attempts failed. Last error:\n{last_error}")

if __name__ == "__main__":
    main()
