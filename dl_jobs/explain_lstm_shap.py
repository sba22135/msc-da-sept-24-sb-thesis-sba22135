#!/usr/bin/env python
import os, json, argparse, glob
import numpy as np, pandas as pd, shap, torch
import matplotlib.pyplot as plt

def load_parquet_folder(folder, max_rows=50_000):
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "**/*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")
    chunks, read = [], 0
    for fp in files:
        df = pd.read_parquet(fp)
        chunks.append(df); read += len(df)
        if read >= max_rows: break
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    return df

def vectorudt_to_numpy(df, label_col="label"):
    if "features" not in df.columns:
        cols = [c for c in df.columns if c != label_col]
        X = df[cols].astype(np.float32).values
        y = df[label_col].astype(np.int64).values
        names = cols
        return X, y, names
    # reconstruct dense features
    feats = df["features"]
    first = next((v for v in feats if isinstance(v, dict) and v.get("size") is not None), None)
    if first is None:
        raise RuntimeError("Could not infer 'features' size from VectorUDT rows.")
    dim = int(first["size"])
    X = np.zeros((len(df), dim), dtype=np.float32)
    for i, v in enumerate(feats):
        if not isinstance(v, dict): continue
        idx = v.get("indices", [])
        val = v.get("values", [])
        if idx is None or val is None: continue
        X[i, np.array(idx, dtype=int)] = np.array(val, dtype=np.float32)
    y = df[label_col].astype(np.int64).values
    names = [f"f{i}" for i in range(dim)]
    return X, y, names

def standardise(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(0, keepdims=True)
    if sd is None:
        sd = X.std(0, keepdims=True) + 1e-6
    return (X - mu) / sd, mu, sd

def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    input_dim = ckpt["input_dim"]
    class FraudLSTM(torch.nn.Module):
        def __init__(self, input_dim, hidden=64, dropout=0.1):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_dim, hidden, batch_first=True)
            self.drop = torch.nn.Dropout(0.1)
            self.fc   = torch.nn.Linear(hidden, 2)
        def forward(self, x):
            x = x.unsqueeze(1)
            _, (h, _) = self.lstm(x)
            h = self.drop(h[-1])
            return self.fc(h)
    m = FraudLSTM(input_dim)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, input_dim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_path", default="models/dl/lstm.pt")
    ap.add_argument("--out_dir",   default="results/figures")
    ap.add_argument("--sample_n", type=int, default=1000)
    ap.add_argument("--background_n", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_parquet_folder(os.path.join(args.data_dir, "test"), max_rows=max(args.sample_n*2, 5000))
    X, y, names = vectorudt_to_numpy(df)
    X, mu, sd = standardise(X)
    # sample foreground + background
    rng = np.random.default_rng(42)
    fg_idx = rng.choice(len(X), size=min(args.sample_n, len(X)), replace=False)
    bg_idx = rng.choice(len(X), size=min(args.background_n, len(X)), replace=False)
    X_fg = X[fg_idx]; X_bg = X[bg_idx]

    model, input_dim = load_model(args.model_path)

    # KernelExplainer on CPU-friendly sizes
    f = lambda data: torch.softmax(model(torch.tensor(data, dtype=torch.float32)), dim=1)[:,1].detach().numpy()
    explainer = shap.KernelExplainer(f, X_bg)
    shap_vals = explainer.shap_values(X_fg, nsamples=200)

    # beeswarm
    plt.figure()
    shap.summary_plot(shap_vals, X_fg, feature_names=names[:X_fg.shape[1]], show=False)
    plt.tight_layout()
    bee = os.path.join(args.out_dir, "shap_summary_beeswarm.png")
    plt.savefig(bee, dpi=200); plt.close()

    # bar
    plt.figure()
    shap.summary_plot(shap_vals, X_fg, plot_type="bar", feature_names=names[:X_fg.shape[1]], show=False)
    plt.tight_layout()
    bar = os.path.join(args.out_dir, "shap_summary_bar.png")
    plt.savefig(bar, dpi=200); plt.close()

    print("Saved SHAP plots:", bee, bar)

if __name__ == "__main__":
    main()
