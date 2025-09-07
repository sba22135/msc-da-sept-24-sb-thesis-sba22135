#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse, glob, numpy as np, pandas as pd, shap, torch
import matplotlib.pyplot as plt
from train_deep import load_parquet_folder, vectorudt_to_numpy, standardise_train, standardise_apply, FraudLSTM, FraudMLP, FraudCNN1D, FraudTransformer

def load_model_any(path):
    ckpt = torch.load(path, map_location="cpu")
    model_name = ckpt.get("model", "mlp")
    in_dim = ckpt["input_dim"]
    if model_name == "lstm": m = FraudLSTM(in_dim)
    elif model_name == "mlp": m = FraudMLP(in_dim)
    elif model_name == "cnn": m = FraudCNN1D(in_dim)
    elif model_name == "transformer": m = FraudTransformer(in_dim)
    else: raise ValueError(f"Unknown model in checkpoint: {model_name}")
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m, model_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", default="results/figures")
    ap.add_argument("--sample_n", type=int, default=1000)
    ap.add_argument("--background_n", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_parquet_folder(os.path.join(args.data_dir, "test"), max_rows=max(args.sample_n*2, 5000))
    X, y, names = vectorudt_to_numpy(df)
    mu, sd = standardise_train(X)
    X = standardise_apply(X, mu, sd)

    rng = np.random.default_rng(42)
    fg_idx = rng.choice(len(X), size=min(args.sample_n, len(X)), replace=False)
    bg_idx = rng.choice(len(X), size=min(args.background_n, len(X)), replace=False)
    X_fg = X[fg_idx]; X_bg = X[bg_idx]

    model, mname = load_model_any(args.model_path)

    f = lambda data: torch.softmax(model(torch.tensor(data, dtype=torch.float32)), dim=1)[:,1].detach().numpy()
    explainer = shap.KernelExplainer(f, X_bg)
    shap_vals = explainer.shap_values(X_fg, nsamples=200)

    bee = os.path.join(args.out_dir, f"shap_{mname}_beeswarm.png")
    bar = os.path.join(args.out_dir, f"shap_{mname}_bar.png")

    plt.figure(); shap.summary_plot(shap_vals, X_fg, feature_names=names[:X_fg.shape[1]], show=False)
    plt.tight_layout(); plt.savefig(bee, dpi=200); plt.close()

    plt.figure(); shap.summary_plot(shap_vals, X_fg, plot_type="bar", feature_names=names[:X_fg.shape[1]], show=False)
    plt.tight_layout(); plt.savefig(bar, dpi=200); plt.close()

    print("Saved SHAP:", bee, bar)

if __name__ == "__main__":
    main()
