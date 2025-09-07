# MSc Thesis — Fraud Detection with Big Data, Deep Learning & XAI

This repository contains the code for my MSc Data Analytics thesis.

## Structure
- `spark_jobs/` — Spark MLlib baselines (LogReg, RF, GBT)
- `deep_learning/` — CNN, LSTM, MLP, Transformer trainers
- `dashboard/` & `app.py` — Streamlit app (model comparison, threshold sweeps, PR curves, SHAP)
- `results/` — metrics JSON/CSVs and small figures (no raw data pushed)
- `scripts/` — helper scripts (run Spark training, run dashboard)
- `figures/` — thesis figures (optional)

## Setup
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Spark baselines (example)
\`\`\`bash
bash scripts/run_spark.sh
\`\`\`

## Dashboard
\`\`\`bash
bash scripts/run_dashboard.sh
\`\`\`

## Notes
- Data and large models are ignored by `.gitignore`.
- SHAP is used for model explainability.
- This repository currently has **no license**; all rights reserved.
