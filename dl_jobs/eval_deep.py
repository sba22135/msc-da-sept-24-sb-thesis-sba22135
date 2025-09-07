#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse, glob
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

# -------- import helpers & model defs from train_deep ----------
from train_deep import (
    load_parquet_folder, vectorudt_to_numpy,
    standardise_train, standardise_apply,
    FraudLSTM, FraudMLP, FraudCNN1D, FraudTransformer
)

torch.set_num_threads(1)

def build_from_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    mname = ckpt.get("model","mlp")
    in_dim = ckpt["input_dim"]
    if mname == "lstm": m = FraudLSTM(in_dim)
    elif mname == "mlp": m = FraudMLP(in_dim)
    elif mname == "cnn": m = FraudCNN1D(in_dim)
    elif mname == "transformer": m = FraudTransformer(in_dim)
    else: raise ValueError(f"Unknown model in checkpoint: {mname}")
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, mname, in_dim

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

def predict_batched(model, X, device, bs=2048):
    probs = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=device)
            pr = torch.softmax(model(xb), dim=1)[:,1].cpu().numpy()
            probs[i:i+bs] = pr
    return probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--train_max_rows", type=int, default=50000, help="to learn standardisation stats")
    ap.add_argument("--test_max_rows", type=int, default=50000)
    ap.add_argument("--eval_bs", type=int, default=2048)
    args = ap.parse_args()

    # Load model
    model, mname, in_dim = build_from_ckpt(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Data: small train slice for standardisation stats + test subset
    train_dir = os.path.join(args.data_dir, "train")
    test_dir  = os.path.join(args.data_dir, "test")

    df_tr_small = load_parquet_folder(train_dir, max_rows=args.train_max_rows)
    df_te = load_parquet_folder(test_dir, max_rows=args.test_max_rows)

    Xtr, ytr, _ = vectorudt_to_numpy(df_tr_small)
    Xte, yte, _ = vectorudt_to_numpy(df_te)

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xte = (Xte - mu) / sd

    # Predict in batches (low memory)
    te_probs = predict_batched(model, Xte, device=device, bs=args.eval_bs)

    # Metrics at t=0.50 and t=0.30
    te_50 = metrics_from_probs(yte, te_probs, 0.50)
    te_30 = metrics_from_probs(yte, te_probs, 0.30)

    # Sweep
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    sweep, best = [], (-1, None, None)
    for t in thresholds:
        acc, pr, rc, f1, prauc, roauc, cm = metrics_from_probs(yte, te_probs, float(t))
        row = {"threshold": float(t), "precision": pr, "recall": rc, "f1": f1, "confusion_matrix": cm}
        sweep.append(row)
        if f1 > best[0]: best = (f1, float(t), row)

    out = {
        "model": mname,
        "settings": {
            "train_max_rows": args.train_max_rows,
            "test_max_rows": args.test_max_rows,
            "eval_bs": args.eval_bs
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
        "test_rows": int(len(df_te))
    }
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics to {args.out_metrics}")
    print(f"Best F1 @ threshold={best[1]} -> {best[0]:.4f}")

if __name__ == "__main__":
    main()
