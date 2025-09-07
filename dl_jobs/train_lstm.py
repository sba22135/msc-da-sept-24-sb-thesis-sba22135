#!/usr/bin/env python
import os, json, argparse, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from collections import Counter

# ---------- IO helpers ----------
def load_parquet_folder(folder, max_rows=200_000, label_col="label"):
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "**/*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")
    chunks, read = [], 0
    for fp in files:
        df = pd.read_parquet(fp)
        if label_col not in df.columns:
            raise RuntimeError(f"'{label_col}' not found in {fp}. Columns: {list(df.columns)}")
        chunks.append(df)
        read += len(df)
        if read >= max_rows:
            break
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
    feats = df["features"]
    first = next((v for v in feats if isinstance(v, dict) and v.get("size") is not None), None)
    if first is None:
        raise RuntimeError("Could not infer 'features' size from VectorUDT rows.")
    dim = int(first["size"])
    X = np.zeros((len(df), dim), dtype=np.float32)
    for i, v in enumerate(feats):
        if not isinstance(v, dict): continue
        idx = v.get("indices", []); val = v.get("values", [])
        if idx is None or val is None: continue
        X[i, np.array(idx, dtype=int)] = np.array(val, dtype=np.float32)
    y = df[label_col].astype(np.int64).values
    names = [f"f{i}" for i in range(dim)]
    return X, y, names

def standardise_train(Xtr):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    return mu, sd

def standardise_apply(X, mu, sd):
    return (X - mu) / sd

# ---------- (optional) oversampling ----------
def oversample_minority(X, y, ratio=1.0, seed=42):
    """
    ratio = desired (minority / majority) after oversampling.
    e.g., if majority=900k, minority=10k (1:90), ratio=0.1 -> make it ~1:10.
    """
    rng = np.random.default_rng(seed)
    cnt = Counter(y)
    maj = 0 if cnt.get(0,0) >= cnt.get(1,0) else 1
    minc = 1 - maj
    n_maj = cnt.get(maj, 0); n_min = cnt.get(minc, 0)
    if n_min == 0 or n_maj == 0: return X, y  # nothing to do
    target_min = int(ratio * n_maj)
    if target_min <= n_min: return X, y
    need = target_min - n_min
    idx_min = np.where(y == minc)[0]
    add_idx = rng.choice(idx_min, size=need, replace=True)
    X_new = np.concatenate([X, X[add_idx]], axis=0)
    y_new = np.concatenate([y, y[add_idx]], axis=0)
    return X_new, y_new

# ---------- models ----------
class FraudLSTM(nn.Module):
    def __init__(self, input_dim, hidden=64, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 2)
    def forward(self, x):
        x = x.unsqueeze(1)  # [B,1,F]
        _, (h, _) = self.lstm(x)
        h = self.drop(h[-1])
        return self.fc(h)

class FraudMLP(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 2)
        )
    def forward(self, x):
        return self.net(x)

# ---------- utils ----------
def to_tensors(X, y): return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def batch_iter(X, Y, bs=256, shuffle=True, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Y))
    if shuffle: rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]
        yield X[j], Y[j]

def metrics_from_probs(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try: auc_roc = roc_auc_score(y_true, y_prob)
    except Exception: auc_roc = None
    try: auc_pr  = average_precision_score(y_true, y_prob)
    except Exception: auc_pr = None
    cm = confusion_matrix(y_true, y_pred).tolist()
    return acc, prec, rec, f1, auc_pr, auc_roc, cm

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_rows", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", choices=["lstm","mlp"], default="lstm")
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--oversample_ratio", type=float, default=0.0, help="Target minority/majority ratio (0=off).")
    ap.add_argument("--out_model", default="models/dl/lstm.pt")
    ap.add_argument("--out_metrics", default="results/dl_lstm_metrics.json")
    args = ap.parse_args()

    # Load data
    train_dir = os.path.join(args.data_dir, "train")
    test_dir  = os.path.join(args.data_dir, "test")
    print("Loading train subset…")
    df_train = load_parquet_folder(train_dir, max_rows=args.max_rows)
    print("Loading test subset…")
    df_test  = load_parquet_folder(test_dir,  max_rows=200_000)

    # split
    df_tr, df_val = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train["label"])

    # to numpy (handles VectorUDT or flat)
    Xtr, ytr, _ = vectorudt_to_numpy(df_tr)
    Xva, yva, _ = vectorudt_to_numpy(df_val)
    Xte, yte, _ = vectorudt_to_numpy(df_test)

    # optional oversampling on TRAIN ONLY
    if args.oversample_ratio and args.oversample_ratio > 0:
        Xtr, ytr = oversample_minority(Xtr, ytr, ratio=args.oversample_ratio)
        print(f"Oversampled minority to ratio {args.oversample_ratio:.3f}. Train size -> {Xtr.shape[0]}")

    # standardise using TRAIN stats
    mu, sd = standardise_train(Xtr)
    Xtr = standardise_apply(Xtr, mu, sd)
    Xva = standardise_apply(Xva, mu, sd)
    Xte = standardise_apply(Xte, mu, sd)

    Xtr_t, ytr_t = to_tensors(Xtr, ytr)
    Xva_t, yva_t = to_tensors(Xva, yva)
    Xte_t, yte_t = to_tensors(Xte, yte)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "lstm":
        net = FraudLSTM(input_dim=Xtr.shape[1]).to(device)
    else:
        net = FraudMLP(input_dim=Xtr.shape[1]).to(device)

    # Class-weighted loss
    if args.use_class_weights:
        pos = int((ytr_t == 1).sum().item())
        neg = int((ytr_t == 0).sum().item())
        pos_weight = max(1.0, neg / max(1, pos))
        class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: [neg=1.0, pos={pos_weight:.2f}]")
    else:
        crit = nn.CrossEntropyLoss()

    opt = optim.Adam(net.parameters(), lr=args.lr)

    # Train
    print("Training…")
    for ep in range(args.epochs):
        net.train()
        total_loss = 0.0
        for xb, yb in batch_iter(Xtr_t.numpy(), ytr_t.numpy(), bs=args.bs, shuffle=True, seed=42+ep):
            xb = torch.tensor(xb, dtype=torch.float32, device=device)
            yb = torch.tensor(yb, dtype=torch.long, device=device)
            opt.zero_grad()
            logits = net(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            total_loss += loss.item()

        # quick validation @ 0.5
        net.eval()
        with torch.no_grad():
            va_probs = torch.softmax(net(Xva_t.to(device)), dim=1)[:,1].cpu().numpy()
            _, va_prec, va_rec, va_f1, va_pr, va_roc, _ = metrics_from_probs(yva_t.numpy(), va_probs, 0.5)
        print(f"Epoch {ep+1}/{args.epochs}  loss={total_loss:.3f}  val_f1={va_f1:.4f}  val_rec={va_rec:.3f}  val_prec={va_prec:.3f}")

    # Test evaluation
    net.eval()
    with torch.no_grad():
        te_probs = torch.softmax(net(Xte_t.to(device)), dim=1)[:,1].cpu().numpy()

    # metrics at t=0.50 and t=0.30 (recall-friendly)
    te_50 = metrics_from_probs(yte_t.numpy(), te_probs, 0.50)
    te_30 = metrics_from_probs(yte_t.numpy(), te_probs, 0.30)

    # threshold sweep & best-F1
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    sweep = []
    best = (-1, None, None)  # (f1, t, row)
    for t in thresholds:
        acc, pr, rc, f1, prauc, roauc, cm = metrics_from_probs(yte_t.numpy(), te_probs, float(t))
        row = {"threshold": float(t), "precision": pr, "recall": rc, "f1": f1, "confusion_matrix": cm}
        sweep.append(row)
        if f1 > best[0]: best = (f1, float(t), row)

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "input_dim": Xtr.shape[1], "model": args.model}, args.out_model)

    out = {
        "model": args.model,
        "settings": {
            "use_class_weights": bool(args.use_class_weights),
            "oversample_ratio": float(args.oversample_ratio),
            "epochs": args.epochs,
            "bs": args.bs,
            "lr": args.lr
        },
        "test_metrics_t0_50": {
            "accuracy": te_50[0], "precision": te_50[1], "recall": te_50[2], "f1": te_50[3],
            "auc_pr": te_50[4], "auc_roc": te_50[5], "confusion_matrix": te_50[6]
        },
        "test_metrics_t0_30": {
            "accuracy": te_30[0], "precision": te_30[1], "recall": te_30[2], "f1": te_30[3],
            "auc_pr": te_30[4], "auc_roc": te_30[5], "confusion_matrix": te_30[6]
        },
        "threshold_sweep": sweep,
        "best_f1": {"f1": best[0], "threshold": best[1], "row": best[2]},
        "train_rows": int(len(df_tr)), "val_rows": int(len(df_val)), "test_rows": int(len(df_test))
    }
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved model to {args.out_model}")
    print(f"Saved metrics to {args.out_metrics}")
    print(f"Best F1 @ threshold={best[1]} -> {best[0]:.4f}")

if __name__ == "__main__":
    main()
