import json, os, numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="FraudLab Dashboard", layout="wide")

st.title("Fraud Detection Results & Explainability")

st.header("Baseline Models (Spark)")
metrics_path = "results/spark_baseline_metrics.json"
test_report_path = "results/test_report.json"

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        valid_metrics = json.load(f)
    st.subheader("Validation Metrics")
    st.json(valid_metrics)
else:
    st.info("Run baseline training to create results/spark_baseline_metrics.json")

if os.path.exists(test_report_path):
    with open(test_report_path) as f:
        test_metrics = json.load(f)
    st.subheader("Test Metrics")
    model = st.selectbox("Choose model", list(test_metrics.keys()))
    st.json(test_metrics[model])

    cm = np.array(test_metrics[model]["confusion_matrix"])
    fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], hoverongaps=False))
    fig.update_layout(title=f"Confusion Matrix: {model}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Run evaluation to create results/test_report.json")

st.header("Explainability")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tree Model (GBTs) — SHAP Kernel (sample)")
    if os.path.exists("results/xai_tree/X_sample.npy"):
        X = np.load("results/xai_tree/X_sample.npy")
        shap_vals = np.load("results/xai_tree/shap_values.npy", allow_pickle=True)
        st.write("Loaded SHAP arrays for tree model (showing summary stats).")
        st.write({"X_shape": X.shape, "shap_shape": np.array(shap_vals).shape})
    else:
        st.info("Run 12_explain_shap_tree.py to generate arrays.")

with col2:
    st.subheader("LSTM — SHAP Kernel & LIME (sample)")
    lime_path = "results/xai_lstm/lime_explanation.txt"
    if os.path.exists(lime_path):
        with open(lime_path) as f:
            st.text(f.read())
    else:
        st.info("Run 12_explain_shap_pytorch.py to generate explanations.")

st.caption("Note: Full SHAP plots are heavy; for the thesis, sample 100–500 rows and include screenshots.")
