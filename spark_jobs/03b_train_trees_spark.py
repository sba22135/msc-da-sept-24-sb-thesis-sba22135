import argparse, os, json
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.sql import functions as F


def get_spark(master):
    return (SparkSession.builder
            .master(master)
            .appName("train_trees_small")
            .config("spark.sql.shuffle.partitions","2")
            .config("spark.default.parallelism","2")
            .getOrCreate())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--train_frac", type=float, default=0.005)  # 0.5% of train
    ap.add_argument("--master", default="local[1]")
    args = ap.parse_args()

    spark = get_spark(args.master)

    train = spark.read.parquet(os.path.join(args.data_dir,"train"))

    pos = train.filter("label = 1").count()
    neg = train.filter("label = 0").count()
    pos_weight = float(neg) / float(max(pos, 1))
    train = train.withColumn(
     "weight",
      F.when(F.col("label") == 1, F.lit(pos_weight)).otherwise(F.lit(1.0))
    )

    valid = spark.read.parquet(os.path.join(args.data_dir,"valid"))

    # tiny sample to keep memory low
    if args.train_frac < 1.0:
        train = train.sample(False, args.train_frac, seed=42)

    # columns expected from our preprocess step
    FEATURES = "features"
    LABEL    = "label"
    WEIGHT_NAME   = "weight"
    weight_col = WEIGHT_NAME  if WEIGHT_NAME  in train.columns else None
    has_w    = weight_col is not None

    os.makedirs(args.models_dir, exist_ok=True)

    # ---------- RandomForest (small) ----------
    rf = RandomForestClassifier(
        featuresCol=FEATURES, labelCol=LABEL,
        weightCol=weight_col if has_w else None,
        numTrees=20, maxDepth=5, subsamplingRate=0.5,
        featureSubsetStrategy="auto", seed=42
    )
    rf_m = rf.fit(train)
    rf_path = os.path.join(args.models_dir, "rf")
    rf_m.write().overwrite().save(rf_path)
    print(f"Saved RF to: {rf_path}")

    # ---------- GBT (small) ----------
    gbt = GBTClassifier(
        featuresCol=FEATURES, labelCol=LABEL,
        weightCol=weight_col if has_w else None,
        maxIter=30, maxDepth=5, stepSize=0.1, subsamplingRate=0.5,
        seed=42
    )
    gbt_m = gbt.fit(train)
    gbt_path = os.path.join(args.models_dir, "gbt")
    gbt_m.write().overwrite().save(gbt_path)
    print(f"Saved GBT to: {gbt_path}")

    spark.stop()

if __name__ == "__main__":
    main()
