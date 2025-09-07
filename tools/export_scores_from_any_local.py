import os, glob, shutil, argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel, GBTClassificationModel
from pyspark.ml.functions import vector_to_array

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="Local Parquet glob (absolute path recommended)")
ap.add_argument("--rf-pipeline", default="models/spark_baselines_10m/rf")
ap.add_argument("--gbt-pipeline", default="models/spark_baselines_10m/gbt")
ap.add_argument("--shap-n", type=int, default=10000)
args = ap.parse_args()

spark = (SparkSession.builder
         .appName("export_scores_from_any_local")
         .config("spark.hadoop.fs.defaultFS","file:///")
         .getOrCreate())

RES = Path("results"); RES.mkdir(exist_ok=True)

def to_file_uris(pattern: str):
    matches = glob.glob(pattern)
    if not matches:
        raise SystemExit(f"[error] no files matched {pattern}")
    return ["file://" + str(Path(m).resolve()) for m in matches]

def read_parquet_local(pattern: str):
    return spark.read.parquet(*to_file_uris(pattern))

def export_scores(pred, model_name):
    sel = pred.select(
        F.col("label").cast("int").alias("label"),
        vector_to_array("probability")[1].alias("score")
    )
    tmp = f"results/_scores_{model_name}_tmp"
    out_csv = RES / f"scores_{model_name}.csv"
    sel.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp)
    part = glob.glob(os.path.join(tmp, "part-*.csv"))[0]
    shutil.move(part, out_csv.as_posix())
    shutil.rmtree(tmp)
    print(f"[ok] wrote {out_csv}")

def export_shap_npz(pred, model_name, n):
    if "features" not in pred.columns: return
    sub = pred.select(vector_to_array("features").alias("x"), F.col("label").alias("y")).limit(n).toPandas()
    import numpy as np
    if len(sub)==0: return
    X = np.stack(sub["x"].values); y = sub["y"].to_numpy()
    np.savez_compressed(RES / f"shap_data_{model_name}.npz", X=X, y=y)
    print(f"[ok] wrote shap_data_{model_name}.npz")

df = read_parquet_local(args.data)

# RF
rfm = RandomForestClassificationModel.load(args.rf_pipeline)
rf_pred = rfm.transform(df)
export_scores(rf_pred,"rf"); export_shap_npz(rf_pred,"rf",args.shap_n)

# GBT
gbtm = GBTClassificationModel.load(args.gbt_pipeline)
gbt_pred = gbtm.transform(df)
export_scores(gbt_pred,"gbt"); export_shap_npz(gbt_pred,"gbt",args.shap_n)

spark.stop()
