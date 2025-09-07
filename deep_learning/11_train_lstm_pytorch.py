import argparse, os, json
import pyarrow.parquet as pq
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from utils.metrics import compute_classification_metrics

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.type_seq = np.stack(df['type_seq'].to_numpy())
        self.amount_seq = np.stack(df['amount_seq'].to_numpy())
        self.old_seq = np.stack(df['oldbalanceOrg_seq'].to_numpy())
        self.new_seq = np.stack(df['newbalanceOrg_seq'].to_numpy())
        self.y = df['label'].astype(np.int64).to_numpy()

        self.mu = self.amount_seq.mean()
        self.sd = self.amount_seq.std() + 1e-6
        self.amount_seq = (self.amount_seq - self.mu) / self.sd
        self.old_seq = (self.old_seq - self.old_seq.mean()) / (self.old_seq.std()+1e-6)
        self.new_seq = (self.new_seq - self.new_seq.mean()) / (self.new_seq.std()+1e-6)

        self.type_seq = self.type_seq.astype(np.int64)
        self.amount_seq = self.amount_seq.astype(np.float32)
        self.old_seq = self.old_seq.astype(np.float32)
        self.new_seq = self.new_seq.astype(np.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (self.type_seq[idx], self.amount_seq[idx], self.old_seq[idx], self.new_seq[idx]), self.y[idx]

class LSTMNet(nn.Module):
    def __init__(self, n_types:int, emb_dim=8, hidden=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_types, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim+3, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, type_seq, amount_seq, old_seq, new_seq):
        e = self.emb(type_seq)
        x = torch.cat([e, amount_seq.unsqueeze(-1), old_seq.unsqueeze(-1), new_seq.unsqueeze(-1)], dim=-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logit = self.head(last)
        return logit.squeeze(-1)

def load_parquet(path, limit=None):
    df = pq.read_table(path).to_pandas()
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=50000, help="Sample limit for quick training")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_parquet(args.seq_path, limit=args.limit)
    from pandas import Series  # ensure pandas types are available
    n_types = int(df["type_seq"].explode().max()) + 1

    from pandas import DataFrame
    from sklearn.model_selection import train_test_split
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_ds = SeqDataset(train_df)
    valid_ds = SeqDataset(valid_df)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LSTMNet(n_types=n_types).to(args.device)
    pos_weight = max(1.0, (len(train_ds.y) - train_ds.y.sum()) / max(train_ds.y.sum(), 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=args.device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = {"epoch": -1, "f1": -1}

    for epoch in range(1, args.epochs+1):
        model.train()
        for (type_seq, amount_seq, old_seq, new_seq), y in train_loader:
            type_seq = torch.tensor(type_seq, dtype=torch.long, device=args.device)
            amount_seq = torch.tensor(amount_seq, dtype=torch.float32, device=args.device)
            old_seq = torch.tensor(old_seq, dtype=torch.float32, device=args.device)
            new_seq = torch.tensor(new_seq, dtype=torch.float32, device=args.device)
            y = torch.tensor(y, dtype=torch.float32, device=args.device)

            optim.zero_grad()
            logits = model(type_seq, amount_seq, old_seq, new_seq)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

        model.eval()
        probs = []
        y_true = []
        with torch.no_grad():
            for (type_seq, amount_seq, old_seq, new_seq), y in valid_loader:
                type_seq = torch.tensor(type_seq, dtype=torch.long, device=args.device)
                amount_seq = torch.tensor(amount_seq, dtype=torch.float32, device=args.device)
                old_seq = torch.tensor(old_seq, dtype=torch.float32, device=args.device)
                new_seq = torch.tensor(new_seq, dtype=torch.float32, device=args.device)
                logits = model(type_seq, amount_seq, old_seq, new_seq)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p.tolist())
                y_true.extend(y.tolist())

        metrics, cm = compute_classification_metrics(np.array(y_true), np.array(probs))
        print(f"Epoch {epoch}: valid f1={metrics['f1']:.4f} auc={metrics['auc']}")

        if metrics["f1"] > best["f1"]:
            best = {"epoch": epoch, "f1": metrics["f1"], "metrics": metrics, "cm": cm}
            torch.save(model.state_dict(), os.path.join(args.out_dir, "lstm.pt"))
            with open(os.path.join(args.out_dir, "valid_metrics.json"), "w") as f:
                json.dump(best, f, indent=2)

    print("Training done. Best:", best)

if __name__ == "__main__":
    main()
