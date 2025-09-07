import argparse, os
import pandas as pd
import numpy as np

TYPES = ["PAYMENT","TRANSFER","CASH_OUT","DEBIT","CASH_IN"]
MERCH = ["grocery","electronics","restaurants","travel","online","atm","utilities","fashion"]

def gen_data(n_rows:int, seed:int=42):
    rng = np.random.default_rng(seed)
    n_accts = max(1000, n_rows // 10)
    senders = [f"C{rng.integers(1, n_accts)}" for _ in range(n_rows)]
    receivers = [f"M{rng.integers(1, int(n_accts*1.2))}" for _ in range(n_rows)]
    steps = rng.integers(1, 744, n_rows)  # hours in ~31 days
    types = rng.choice(TYPES, n_rows, p=[0.45,0.12,0.28,0.05,0.10])
    amounts = np.round(rng.gamma(shape=2.0, scale=200.0, size=n_rows), 2)
    amount_boost = (types=="TRANSFER")*rng.uniform(1.0,2.5,n_rows) + (types=="CASH_OUT")*rng.uniform(1.0,2.0,n_rows)
    amounts = np.round(amounts * (1.0 + amount_boost*0.3), 2)

    old_org = np.round(rng.normal(loc=1500, scale=800, size=n_rows).clip(0, 10000),2)
    new_org = (old_org - amounts).clip(0, None)
    old_dst = np.round(rng.normal(loc=1000, scale=600, size=n_rows).clip(0, 20000),2)
    new_dst = np.round(old_dst + amounts, 2)

    base_p = 0.012  # ~1.2%
    p = np.full(n_rows, base_p)
    p += (types=="TRANSFER")*0.01
    p += (types=="CASH_OUT")*0.006
    p += (amounts>1500)*0.006
    p += (new_org==0)*0.005
    is_fraud = rng.binomial(1, np.clip(p, 0, 0.25))

    is_flagged = ((amounts>200000).astype(int))

    merch = rng.choice(MERCH, n_rows)
    df = pd.DataFrame(dict(
        step=steps,
        type=types,
        amount=amounts,
        nameOrig=senders,
        oldbalanceOrg=old_org,
        newbalanceOrg=new_org,
        nameDest=receivers,
        oldbalanceDest=old_dst,
        newbalanceDest=new_dst,
        merchant=merch,
        isFraud=is_fraud,
        isFlaggedFraud=is_flagged
    ))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--n_rows", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = gen_data(args.n_rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  rows={len(df)}  fraud_rate={df['isFraud'].mean():.4f}")

if __name__ == "__main__":
    main()
