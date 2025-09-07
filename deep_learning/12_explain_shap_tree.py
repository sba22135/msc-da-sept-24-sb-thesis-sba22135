import argparse, os, json, numpy as np
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.linalg import DenseVector
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def get_spark():
    return SparkSession.builder.appName("FraudLab-XAI-Tree").getOrCreate()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_size", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    spark = get_spark()
    valid = spark.read.parquet(os.path.join(args.data_dir, "valid")).limit(args.sample_size)

    rows = valid.select("features","label").toLocalIterator()
    X, y = [], []
    for r in rows:
        X.append(np.array(r["features"]))
        y.append(int(r["label"]))
    X = np.stack(X)
    y = np.array(y)

    import shap
    model = GBTClassificationModel.load(os.path.join(args.models_dir, "gbt"))

    def predict_proba(arr):
        pdf = []
        for i in range(0, len(arr), 512):
            batch = arr[i:i+512]
            sdf = spark.createDataFrame([(DenseVector(v.tolist()),) for v in batch], ["features"])
            pr = model.transform(sdf).select("probability").toPandas()
            pdf.append(np.stack(pr["probability"].apply(lambda v: np.array(v)).values))
        proba = np.vstack(pdf)
        return proba

    bg = X[np.random.choice(len(X), size=min(100, len(X)), replace=False)]
    explainer = shap.KernelExplainer(lambda z: predict_proba(z)[:,1], bg)
    shap_values = explainer.shap_values(X[:200], nsamples=100)
    np.save(os.path.join(args.out_dir, "shap_values.npy"), shap_values)
    np.save(os.path.join(args.out_dir, "X_sample.npy"), X[:200])
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"note":"KernelExplainer on GBT (sampled)"}, f, indent=2)
    print("Saved SHAP arrays to:", args.out_dir)

if __name__ == "__main__":
    main()
