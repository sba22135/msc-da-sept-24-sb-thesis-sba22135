def get_spark(app_name="FraudLab", master=None):
    from pyspark.sql import SparkSession
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

import argparse, os
from pyspark.sql import functions as F, Window
from pyspark.sql.types import ArrayType, IntegerType, DoubleType

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw CSV (local/HDFS) with PaySim-like columns")
    ap.add_argument("--out", required=True, help="Output parquet of sequences")
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--master", default=None)
    args = ap.parse_args()

    spark = get_spark("FraudLab-SeqBuild", master=args.master)
    df = spark.read.csv(args.input, header=True, inferSchema=True)

    cols = ["nameOrig","step","type","amount","oldbalanceOrg","newbalanceOrg","isFraud"]
    df = df.select(*cols)
    w = Window.partitionBy("nameOrig").orderBy(F.col("step").asc())
    df = df.withColumn("rn", F.row_number().over(w))

    types = [r[0] for r in df.select("type").distinct().collect()]
    type_map = {t:i for i,t in enumerate(sorted(types))}
    type_map_bc = spark.sparkContext.broadcast(type_map)
    map_udf = F.udf(lambda t: int(type_map_bc.value.get(t,0)), "int")
    df = df.withColumn("type_idx", map_udf(F.col("type")))

    seq_len = args.seq_len
    df = df.withColumn("win_id", ((F.col("rn")-1)/seq_len).cast("int"))
    agg = df.groupBy("nameOrig","win_id").agg(
        F.collect_list("type_idx").alias("type_seq"),
        F.collect_list("amount").alias("amount_seq"),
        F.collect_list("oldbalanceOrg").alias("oldbalanceOrg_seq"),
        F.collect_list("newbalanceOrg").alias("newbalanceOrg_seq"),
        F.max("isFraud").alias("label")
    )
    trim_i = F.udf(lambda a: a[-seq_len:], ArrayType(IntegerType()))
    trim_f = F.udf(lambda a: a[-seq_len:], ArrayType(DoubleType()))
    filt = (agg.where(F.size("type_seq") >= seq_len)
            .withColumn("type_seq", trim_i("type_seq"))
            .withColumn("amount_seq", trim_f("amount_seq"))
            .withColumn("oldbalanceOrg_seq", trim_f("oldbalanceOrg_seq"))
            .withColumn("newbalanceOrg_seq", trim_f("newbalanceOrg_seq"))
           )
    filt.write.mode("overwrite").parquet(args.out)
    print(f"Wrote sequences to: {args.out}")

if __name__ == "__main__":
    main()
