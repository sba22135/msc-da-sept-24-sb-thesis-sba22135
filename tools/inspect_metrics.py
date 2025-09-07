import json
from pathlib import Path

BASE = Path("results/spark_baseline_metrics_10m.json")
TEST = Path("results/spark_test_metrics_10m.json")

def load_json(p: Path, label: str):
    if not p.exists():
        print(f"[WARN] {label} file not found at {p}")
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[ERROR] Failed to parse {label}: {e}")
        return {}

baseline = load_json(BASE, "baseline")
test = load_json(TEST, "test")

def normalize(metrics_dict):
    rows = []
    for model, m in (metrics_dict or {}).items():
        if not isinstance(m, dict): 
            continue
        row = {"model": model}
        def add_numeric(d):
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    row[k.lower()] = v
        add_numeric(m)
        if isinstance(m.get("metrics"), dict):
            add_numeric(m["metrics"])
        rows.append(row)
    return rows

def to_table(rows):
    if not rows:
        return "  (no rows)\n"
    keys = ["model"] + sorted({k for r in rows for k in r.keys() if k != "model"})
    widths = {k:max(len(k), max(len(f"{r.get(k,'')}") for r in rows)) for k in keys}
    header = " | ".join(k.ljust(widths[k]) for k in keys)
    sep    = "-+-".join("-"*widths[k] for k in keys)
    lines = []
    for r in rows:
        line = " | ".join(f"{r.get(k,'')}".ljust(widths[k]) for k in keys)
        lines.append(line)
    return header+"\n"+sep+"\n"+"\n".join(lines)+"\n"

print("\n=== BASELINE METRICS (train/val) ===")
print(to_table(normalize(baseline)))

print("=== TEST METRICS (holdout) ===")
print(to_table(normalize(test)))

expected = {"logreg","rf","gbt"}
present  = {r['model'] for r in normalize(test)}
missing  = sorted(list(expected - present))
if missing:
    print(f"[NOTE] Missing in test metrics: {', '.join(missing)}")
