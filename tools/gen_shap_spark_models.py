#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP plots for Spark RF & GBT via sklearn surrogates (robust to SHAP API quirks).

- Uses the modern Explainer API and slices to the positive class if needed.
- Beeswarm via shap.plots.beeswarm (2D Explanation).
- Bar chart drawn with matplotlib from mean(|SHAP|) (avoids SHAP bar edge cases).

Inputs: results/shap_data_rf.npz, results/shap_data_gbt.npz
Outputs:
  results/figures/shap_rf_beeswarm.png, shap_rf_bar.png
  results/figures/shap_gbt_beeswarm.png, shap_gbt_bar.png
"""
from pathlib import Path
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import check_random_state

RES = Path("results")
FIG = (RES / "figures"); FIG.mkdir(parents=True, exist_ok=True)
RNG = check_random_state(42)

def load_npz(name):
    p = RES / f"shap_data_{name}.npz"
    if not p.exists():
        print(f"[skip] {name}: {p} not found"); return None, None
    z = np.load(p)
    return z["X"], z["y"]

def make_background(X, n=1000):
    n = min(n, len(X))
    idx = RNG.choice(len(X), size=n, replace=False)
    return X[idx]

def ensure_2d_positive_class(exp):
    """Return a 2-D Explanation for beeswarm/bar by picking positive class."""
    vals = exp.values
    if getattr(vals, "ndim", 2) == 2:
        return exp
    if vals.ndim == 3:  # (n_samples, n_classes, n_features)
        cls = 1 if vals.shape[1] > 1 else 0
        try:
            return exp[:, cls]
        except Exception:
            base = exp.base_values
            if np.ndim(base) == 2:
                base = base[:, cls]
            return shap.Explanation(
                values=vals[:, cls, :],
                base_values=base,
                data=exp.data,
                feature_names=exp.feature_names,
            )
    raise ValueError("Unexpected SHAP values shape; expected 2-D or 3-D.")

def beeswarm_and_bar(exp, prefix, top_n=20):
    # Ensure 2-D Explanation
    exp2d = ensure_2d_positive_class(exp)

    # Beeswarm (works with Explanation 2-D)
    shap.plots.beeswarm(exp2d, show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(FIG / f"shap_{prefix}_beeswarm.png", bbox_inches="tight", dpi=160)
    plt.close()

    # Robust bar: mean(|SHAP|) per feature using matplotlib
    vals = np.asarray(exp2d.values, dtype=float)
    vals = np.nan_to_num(vals)  # guard against NaNs
    mean_abs = np.nanmean(np.abs(vals), axis=0)  # (n_features,)

    # Feature names (fallback to f0..fN-1)
    names = exp2d.feature_names
    if names is None:
        names = [f"f{i}" for i in range(len(mean_abs))]

    # Top-N
    idx = np.argsort(mean_abs)[-top_n:]
    names_top = [names[i] for i in idx]
    vals_top = mean_abs[idx]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(names_top, vals_top)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} features by |SHAP| — {prefix.upper()}")
    plt.tight_layout()
    plt.savefig(FIG / f"shap_{prefix}_bar.png", bbox_inches="tight", dpi=160)
    plt.close()
    print(f"[ok] wrote shap_{prefix}_beeswarm.png and shap_{prefix}_bar.png")

def run_for(model_code, fit_fn):
    X, y = load_npz(model_code)
    if X is None or not X.size:
        return
    # sample for speed
    idx = RNG.choice(len(X), size=min(10000, len(X)), replace=False)
    Xb, yb = X[idx], y[idx]
    model = fit_fn(Xb, yb)
    bg = make_background(Xb, 1000)
    expl = shap.Explainer(model, bg, algorithm="tree")
    exp = expl(Xb)
    beeswarm_and_bar(exp, model_code)

def fit_rf(X, y):
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    ).fit(X, y)

def fit_gbt(X, y):
    return GradientBoostingClassifier(random_state=42).fit(X, y)

if __name__ == "__main__":
    run_for("rf", fit_rf)
    run_for("gbt", fit_gbt)
