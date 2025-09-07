import argparse, os, json, numpy as np, pyarrow.parquet as pq
import shap, lime.lime_tabular as lime_tab
import torch
from 11_train_lstm_pytorch import LSTMNet, SeqDataset, load_parquet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--seq_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_size", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_parquet(args.seq_path, limit=args.sample_size)
    ds = SeqDataset(df)

    n_types = int(df["type_seq"].explode().max()) + 1
    model = LSTMNet(n_types=n_types).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "lstm.pt"), map_location=args.device))
    model.eval()

    X = np.hstack([ds.type_seq, ds.amount_seq, ds.old_seq, ds.new_seq]).astype(np.float32)

    def predict_proba_tab(x):
        T = ds.amount_seq.shape[1]
        splits = np.split(x, indices_or_sections=[T, 2*T, 3*T], axis=1)
        type_seq = splits[0].astype(np.int64)
        amount_seq = splits[1].astype(np.float32)
        old_seq = splits[2].astype(np.float32)
        new_seq = splits[3].astype(np.float32)
        with torch.no_grad():
            import torch as _torch
            logits = model(
                _torch.tensor(type_seq, dtype=_torch.long, device=args.device),
                _torch.tensor(amount_seq, dtype=_torch.float32, device=args.device),
                _torch.tensor(old_seq, dtype=_torch.float32, device=args.device),
                _torch.tensor(new_seq, dtype=_torch.float32, device=args.device),
            )
            p = _torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1-p, p]).T

    bg_idx = np.random.choice(len(X), size=min(100, len(X)), replace=False)
    explainer = shap.KernelExplainer(lambda z: predict_proba_tab(z)[:,1], X[bg_idx])
    shap_values = explainer.shap_values(X[:200], nsamples=100)
    np.save(os.path.join(args.out_dir, "shap_values.npy"), shap_values)
    np.save(os.path.join(args.out_dir, "X_sample.npy"), X[:200])
    print("Saved LSTM SHAP arrays")

    explainer_lime = lime_tab.LimeTabularExplainer(X, feature_names=[f"f{i}" for i in range(X.shape[1])], class_names=["legit","fraud"], discretize_continuous=True)
    exp = explainer_lime.explain_instance(X[0], lambda z: predict_proba_tab(z), num_features=10)
    with open(os.path.join(args.out_dir, "lime_explanation.txt"), "w") as f:
        f.write(str(exp.as_list()))
    print("Saved LIME explanation")

if __name__ == "__main__":
    main()
