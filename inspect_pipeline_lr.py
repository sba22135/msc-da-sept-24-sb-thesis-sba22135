from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

# <- replace with the actual path you find in step 1
PIPE_PATH = Path("/home/hduser/Documents/fraudlab/models/spark_baselines_10m/baseline_pipeline").resolve()

def file_uri(p: Path) -> str:
    return "file://" + str(p)

spark = (SparkSession.builder
         .master("local[1]")
         .appName("inspect-pipeline-lr")
         .config("spark.hadoop.fs.defaultFS","file:///")
         .getOrCreate())

pm = PipelineModel.load(file_uri(PIPE_PATH))

print("Pipeline stages:", len(pm.stages))
for i, st in enumerate(pm.stages):
    print(f"Stage {i}: {type(st).__name__}  uid={st.uid}")
    if isinstance(st, LogisticRegressionModel):
        print("\n=== Logistic Regression Hyperparameters (from saved model) ===")
        for p in st.params:
            try:
                print(f"{p.name} = {st.getOrDefault(p)}")
            except:
                pass

spark.stop()
