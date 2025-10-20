import re
from pathlib import Path
import pandas as pd

# Where your backtest outputs were saved by backtest.py
OUT_DIRS = ["data/backtest"]  
paths = []
for d in OUT_DIRS:
    p = Path(d)
    if p.exists():
        paths += list(p.glob("metrics_*.csv"))

if not paths:
    raise FileNotFoundError("No metrics_*.csv found in OUT_DIRS.")

# Aggregate all metrics files
rows = []
pat = re.compile(r"metrics_(?P<task>regression|classification)_(?P<model>[a-z0-9]+)(?P<ls>_ls)?_top(?P<k>\d+)\.csv")

# Process each metrics file
for fp in paths:
    # Parse metadata from filename
    m = pat.match(fp.name)
    if not m:
        # Skip files that don't follow the naming convention
        continue
    meta = m.groupdict()
    meta["ls"] = bool(meta["ls"])
    meta["k"] = int(meta["k"])

    df = pd.read_csv(fp)
    
    # Expect a single-row metrics file; if multiple, take the first
    rec = df.iloc[0].to_dict()
    rec.update({
        "config": fp.stem.replace("metrics_", ""), 
        "task": meta["task"],
        "model": meta["model"],
        "long_short": meta["ls"],
        "top_k": meta["k"],
        "source_file": str(fp),
    })
    rows.append(rec)

leader = pd.DataFrame(rows)

# Ensure expected columns exist even if some runs missed them
for col in ["Sharpe","CAGR","MaxDD","Vol_ann","Hit_ratio","Turnover_avg","n_months"]:
    if col not in leader.columns:
        leader[col] = pd.NA

# Sort: best Sharpe, then higher CAGR, then smaller (less negative) MaxDD
leader_sorted = leader.sort_values(
    by=["Sharpe","CAGR","MaxDD"],
    ascending=[False, False, True],
    na_position="last"
).reset_index(drop=True)

# Save leaderboard
out_path = Path("data/backtest/leaderboard.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
leader_sorted.to_csv(out_path, index=False)

print(f"Saved leaderboard -> {out_path}")