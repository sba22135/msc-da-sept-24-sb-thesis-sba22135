#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

MODELS = [m.lower() for m in sys.argv[1:]] or ["rf","gbt"]
RES = Path("results"); RES.mkdir(exist_ok=True)

def sweep_for(model):
    csv = RES / f"scores_{model}.csv"
    if not csv.exists():
        print(f"[skip] {model}: {csv} not found"); return
    df = pd.read_csv(csv)
    y = df["label"].astype(int).to_numpy()
    s = df["score"].astype(float).to_numpy()
    grid = np.linspace(0.0, 1.0, 51)
    rows=[]
    for t in grid:
        pred = (s >= t).astype(int)
        tp = int(((pred==1)&(y==1)).sum())
        fp = int(((pred==1)&(y==0)).sum())
        fn = int(((pred==0)&(y==1)).sum())
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
        rows.append({"threshold":float(t),
                     "precision":float(prec),
                     "recall":float(rec),
                     "f1":float(f1)})
    out = {"model":model.upper(),"sweep":rows}
    out_path = RES / f"threshold_sweep_{model}.json"
    out_path.write_text(json.dumps(out))
    print(f"[ok] wrote {out_path}")

if __name__=="__main__":
    for m in MODELS: sweep_for(m)
