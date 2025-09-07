def get_spark(app_name="FraudLab", master=None):
    from pyspark.sql import SparkSession
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

import argparse, os, json
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV path (local) or HDFS path")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--use_hdfs", type=int, default=0)
    ap.add_argument("--master", default=None, help="Spark master, e.g., local[*]")
    args = ap.parse_args()

    spark = get_spark("FraudLab-Preprocess", master=args.master)
    df = spark.read.csv(args.input, header=True, inferSchema=True)

    df = df.dropna(subset=["type","amount","isFraud"])
    df = df.withColumn("label", F.col("isFraud").cast("int"))

    cat_cols = ["type","merchant"]
    idx_cols = [c+"_idx" for c in cat_cols]
    ohe_cols = [c+"_ohe" for c in cat_cols]

    indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in cat_cols]
    encoders = [OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols, handleInvalid="keep")]

    num_cols = ["step","amount","oldbalanceOrg","newbalanceOrg","oldbalanceDest","newbalanceDest"]
    assembler_num = VectorAssembler(inputCols=num_cols, outputCol="num_vec")
    scaler = StandardScaler(inputCol="num_vec", outputCol="num_scaled")

    feature_cols = ohe_cols + ["num_scaled"]
    assembler_all = VectorAssembler(inputCols=feature_cols, outputCol="features")

    stages = indexers + encoders + [assembler_num, scaler, assembler_all]
    pipe = Pipeline(stages=stages)
    model = pipe.fit(df)
    ds = model.transform(df)

    stats = ds.groupBy("label").count().collect()
    counts = {r["label"]: r["count"] for r in stats}
    neg = counts.get(0, 1)
    pos = counts.get(1, 1)
    pos_weight = float(neg / max(pos,1))
    ds = ds.withColumn("weight", F.when(F.col("label")==1, F.lit(pos_weight)).otherwise(F.lit(1.0)))

    train, valid, test = ds.randomSplit([0.7, 0.15, 0.15], seed=42)

    for name, part in [("train",train),("valid",valid),("test",test)]:
        outp = os.path.join(args.out_dir, name)
        part.select("features","label","weight").write.mode("overwrite").parquet(outp)
        print(f"Wrote: {outp}")

    meta = dict(pos=pos, neg=neg, pos_weight=pos_weight, num_cols=num_cols, cat_cols=cat_cols)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta.json")

if __name__ == "__main__":
    main()
