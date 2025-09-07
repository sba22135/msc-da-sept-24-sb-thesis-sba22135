from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegressionModel, RandomForestClassificationModel, GBTClassificationModel
)

# Absolute local dir where models are saved
BASE_LOCAL = Path("/home/hduser/Documents/fraudlab/models/spark_baselines_10m").resolve()

def file_uri(p: Path) -> str:
    # Spark wants a URI when HDFS is default; prefix with file://
    return "file://" + str(p)

spark = (
    SparkSession.builder
    .master("local[1]")
    .appName("inspect")
    .config("spark.hadoop.fs.defaultFS", "file:///")  # force local FS
    .getOrCreate()
)

def show(model, name):
    print(f"\n=== {name} Parameters ===")
    for p in model.params:
        try:
            print(f"{p.name} = {model.getOrDefault(p)}")
        except:
            pass

# Try to load locally
try:
    lr = LogisticRegressionModel.load(file_uri(BASE_LOCAL / "logreg"))
    show(lr, "Logistic Regression")
except Exception as e:
    print("No LR (local):", e)

try:
    rf = RandomForestClassificationModel.load(file_uri(BASE_LOCAL / "rf"))
    show(rf, "Random Forest")
except Exception as e:
    print("No RF (local):", e)

try:
    gbt = GBTClassificationModel.load(file_uri(BASE_LOCAL / "gbt"))
    show(gbt, "Gradient Boosted Trees")
except Exception as e:
    print("No GBT (local):", e)

spark.stop()
