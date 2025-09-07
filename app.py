#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============ Imports ============
import os, json, glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

# ============ Page config & light CSS ============
st.set_page_config(page_title="Fraud Detection – Model Dashboard", layout="wide")
st.markdown("""
<style>
div.block-container {padding-top: 1.5rem; padding-bottom: 1.2rem; max-width: 1200px;}
[data-testid="stMetric"] {margin-bottom: 0.75rem;}
table td, table th {padding: .35rem .5rem;}
</style>
""", unsafe_allow_html=True)

def gap(px=12):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

# ============ Friendly names ============
MODEL_NAME_MAP = {
    "LOGREG": "Logistic Regression",
    "RF": "Random Forest",
    "GBT": "Gradient Boosted Trees",
    "CNN": "Convolutional Neural Network",
    "LSTM": "Long Short-Term Memory",
    "MLP": "Multi-Layer Perceptron",
    "TRANSFORMER": "Transformer",
}

# ============ Paths ============
RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "final_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============ IO helpers ============
@st.cache_data(show_spinner=False)
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_all_results():
    data = {}
    p = os.path.join(RESULTS_DIR, "spark_test_metrics_10m.json")
    if os.path.isfile(p):
        data["SPARK"] = load_json(p)
    for fp in sorted(glob.glob(os.path.join(RESULTS_DIR, "dl_*_metrics.json"))):
        try:
            jd = load_json(fp)
            name = (jd.get("model") or os.path.basename(fp).split("_")[1]).upper()
            data[name] = jd
        except Exception:
            pass
    for fp in sorted(glob.glob(os.path.join(RESULTS_DIR, "threshold_sweep_*.json"))):
        key = "SWEEP_" + os.path.basename(fp).split("_")[-1].split(".")[0].upper()
        data[key] = load_json(fp)
    return data

def dl_to_row(jd):
    t50 = jd.get("test_metrics_t0_50", {}) or {}
    best = jd.get("best_f1", {})
    return dict(
        model=jd.get("model","?").upper(),
        threshold=best.get("threshold", "—"),
        accuracy=t50.get("accuracy"),
        precision=t50.get("precision"),
        recall=t50.get("recall"),
        f1=t50.get("f1"),
        pr_auc=t50.get("auc_pr"),
        roc_auc=t50.get("auc_roc"),
        source="DL"
    )

def spark_to_rows(jd):
    rows=[]
    for m, blob in jd.items():
        if m not in ("logreg","rf","gbt"):
            continue
        metrics = blob.get("metrics", {})
        rows.append(dict(
            model=m.upper(),
            threshold="0.50",
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1=metrics.get("f1"),
            pr_auc=metrics.get("auc_pr"),
            roc_auc=metrics.get("auc_roc"),
            source="SPARK"
        ))
    return rows

def pretty_float(x):
    if x is None:
        return "—"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

# ============ Plot helpers ============
def plot_threshold_sweep(th, prec, rec, f1, title="Threshold sweep"):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(th, prec, label="Precision")
    ax.plot(th, rec, label="Recall")
    ax.plot(th, f1, label="F1")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=.3)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

def plot_pr_curve(rec, prec, title="Precision–Recall (from sweep)"):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(rec, prec, marker="o", linewidth=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    st.pyplot(fig)

def cm_altair_from_any(cm, title:str):
    arr = np.array(cm, dtype=int)
    diag1 = arr[0,0] + arr[1,1]
    diag2 = arr[0,1] + arr[1,0]
    if diag1 >= diag2:
        tn, fp, fn, tp = arr[0,0], arr[0,1], arr[1,0], arr[1,1]
    else:
        tp, fp, fn, tn = arr[0,0], arr[0,1], arr[1,0], arr[1,1]

    df = pd.DataFrame([
        {"True":"1", "Pred":"1", "Cell":"TP", "Count": int(tp)},
        {"True":"1", "Pred":"0", "Cell":"FN", "Count": int(fn)},
        {"True":"0", "Pred":"1", "Cell":"FP", "Count": int(fp)},
        {"True":"0", "Pred":"0", "Cell":"TN", "Count": int(tn)},
    ])
    base = alt.Chart(df).encode(
        x=alt.X("Pred:N", title="Predicted", sort=["0","1"]),
        y=alt.Y("True:N", title="True",     sort=["1","0"])
    )
    heat = base.mark_rect(cornerRadius=6).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="tealblues"), legend=None)
    )
    text = base.mark_text(fontWeight="bold").encode(
        text=alt.Text("Count:Q", format=",.0f")
    )
    labels = base.mark_text(dy=16, opacity=0.9).encode(text="Cell:N")
    return (heat + text + labels).properties(width=480, height=320, title=title)

def find_shap_images(model_name: str):
    base = os.path.join(RESULTS_DIR, "figures")
    bee_candidates = [
        os.path.join(base, f"shap_{model_name.lower()}_beeswarm.png"),
        os.path.join(base, "shap_summary_beeswarm.png"),
    ]
    bar_candidates = [
        os.path.join(base, f"shap_{model_name.lower()}_bar.png"),
        os.path.join(base, "shap_summary_bar.png"),
    ]
    bee = next((p for p in bee_candidates if os.path.isfile(p)), None)
    bar = next((p for p in bar_candidates if os.path.isfile(p)), None)
    return bee, bar

# ============ UI Header ============
st.title("Fraud Detection – Model Comparison & XAI")
st.caption("Test metrics at default decision threshold = 0.50 (unless changed)")
st.markdown("---")

# ============ Load data ============
data = load_all_results()
if not data:
    st.warning("No results found in ./results. Run your training/evaluation jobs first.")
    st.stop()

# ============ Overview table ============
rows=[]
if "SPARK" in data:
    rows += spark_to_rows(data["SPARK"])
for k, v in data.items():
    if k in ("SPARK",) or k.startswith("SWEEP_"):
        continue
    if isinstance(v, dict) and v.get("model"):
        rows.append(dl_to_row(v))
df = pd.DataFrame(rows)

if not df.empty:
    def format_thr(x):
        if isinstance(x, (int, float, np.floating)):
            try:
                return f"{float(x):.2f}"
            except Exception:
                return str(x)
        return str(x) if x is not None else "—"
    if "threshold" in df.columns:
        df["threshold"] = df["threshold"].apply(format_thr)

    df_disp = df.copy()
    for c in ["accuracy","precision","recall","f1","pr_auc","roc_auc"]:
        if c in df_disp.columns:
            df_disp[c] = df_disp[c].apply(pretty_float)

    # replace short codes with friendly names for table
    df_disp["model"] = df_disp["model"].apply(lambda m: MODEL_NAME_MAP.get(m, m))

    st.markdown("##### Model comparison — test metrics (default threshold 0.50)")
    st.dataframe(df_disp, use_container_width=True, height=280)
else:
    st.info("No metrics loaded yet.")

st.markdown("---")
gap(8)

# ============ Sidebar ============
model_names = []
if "SPARK" in data:
    for m in ["logreg","rf","gbt"]:
        if m in data["SPARK"]:
            model_names.append(m.upper())
for k, v in data.items():
    if k not in ("SPARK",) and not k.startswith("SWEEP_") and isinstance(v, dict) and v.get("model"):
        model_names.append(v["model"].upper())
model_names = sorted(list(dict.fromkeys(model_names)))

st.sidebar.header("Controls")
display_names = [MODEL_NAME_MAP.get(m, m) for m in model_names]
display_to_code = {MODEL_NAME_MAP.get(m, m): m for m in model_names}
picked_display = st.sidebar.selectbox("Select a model", options=display_names)
picked = display_to_code[picked_display]

# ============ Details ============
st.subheader(f"Details — {MODEL_NAME_MAP.get(picked, picked)}")

# ---- Spark models ----
if picked in ("LOGREG","RF","GBT") and "SPARK" in data:
    blob = data["SPARK"][picked.lower()]
    metrics = blob.get("metrics",{})
    cm = blob.get("confusion_matrix")

    c1, c2, c3 = st.columns([2,2,3])
    with c1:
        st.metric("Accuracy",  pretty_float(metrics.get("accuracy")))
        st.metric("Precision", pretty_float(metrics.get("precision")))
    with c2:
        st.metric("Recall",    pretty_float(metrics.get("recall")))
        st.metric("F1",        pretty_float(metrics.get("f1")))
    with c3:
        st.metric("PR-AUC",    pretty_float(metrics.get("auc_pr")))
        st.metric("ROC-AUC",   pretty_float(metrics.get("auc_roc")))

    gap(6)
    if cm:
        st.altair_chart(
            cm_altair_from_any(cm, title=f"{MODEL_NAME_MAP.get(picked,picked)} — Confusion Matrix (threshold = 0.50)"),
            use_container_width=False
        )
    st.markdown("---"); gap(12)

    key = "SWEEP_"+picked
    if key in data:
        sw = data[key].get("sweep", [])
        if sw:
            th = [r["threshold"] for r in sw]
            prec = [r["precision"] for r in sw]
            rec  = [r["recall"] for r in sw]
            f1   = [r["f1"] for r in sw]
            plot_threshold_sweep(th, prec, rec, f1, f"Threshold sweep — {MODEL_NAME_MAP.get(picked,picked)}")
            st.markdown("---"); gap(12)
            plot_pr_curve(rec, prec, f"Precision–Recall — {MODEL_NAME_MAP.get(picked,picked)}")

# ---- Deep learning models ----
else:
    jd = None
    for k, v in data.items():
        if k in ("SPARK",) or k.startswith("SWEEP_"): continue
        if isinstance(v, dict) and v.get("model","").upper() == picked:
            jd = v; break
    if jd is None:
        st.info("No details available.")
    else:
        t50 = jd.get("test_metrics_t0_50", {})
        cm = t50.get("confusion_matrix")

        c1, c2, c3 = st.columns([2,2,3])
        with c1:
            st.metric("Accuracy",  pretty_float(t50.get("accuracy")))
            st.metric("Precision", pretty_float(t50.get("precision")))
        with c2:
            st.metric("Recall",    pretty_float(t50.get("recall")))
            st.metric("F1",        pretty_float(t50.get("f1")))
        with c3:
            st.metric("PR-AUC",    pretty_float(t50.get("auc_pr")))
            st.metric("ROC-AUC",   pretty_float(t50.get("auc_roc")))

        gap(6)
        if cm:
            st.altair_chart(
                cm_altair_from_any(cm, title=f"{MODEL_NAME_MAP.get(picked,picked)} — Confusion Matrix (threshold = 0.50)"),
                use_container_width=False
            )
        st.markdown("---"); gap(12)

        sw = jd.get("threshold_sweep", [])
        if sw:
            th = [r["threshold"] for r in sw]
            prec = [r["precision"] for r in sw]
            rec  = [r["recall"] for r in sw]
            f1   = [r["f1"] for r in sw]
            plot_threshold_sweep(th, prec, rec, f1, f"Threshold sweep — {MODEL_NAME_MAP.get(picked,picked)}")
            st.markdown("---"); gap(12)
            plot_pr_curve(rec, prec, f"Precision–Recall — {MODEL_NAME_MAP.get(picked,picked)}")

        bee, bar = find_shap_images(picked)
        imgs = []
        if bee: imgs.append(("SHAP Beeswarm", bee))
        if bar: imgs.append(("SHAP Bar (mean |SHAP|)", bar))
        if imgs:
            st.markdown("---"); gap(12)
            st.markdown("### Explainability (SHAP)")
            cols = st.columns(len(imgs))
            for (label, path), c in zip(imgs, cols):
                with c:
                    st.image(path, caption=label, use_container_width=True)
        else:
            st.caption("No SHAP plots found for this model .")

# ============ Footer ============
st.markdown("---")
st.caption("Note: In imbalanced problems like fraud, accuracy can be deceptive. Judge models by PR-AUC, recall, precision, and F1 (at a sensible threshold).")
