"""
Real-data pipeline, Part 0: reproducible stratified sampling.

Source: Kaggle "Fraud Detection Dataset" (amanalisiddiqui), the public
PaySim synthetic-financial-transactions dataset (Lopez-Rojas, Elmir &
Axelsson, 2016) — 6,362,620 transaction rows, 8,213 fraud (0.1291%).

Per project scope ("use a quarter of the data"), this script draws a
STRATIFIED 25% sample rather than the first 25% of rows by time-step, so
that the true fraud rate (0.1291%) is preserved exactly rather than
depending on which time window happened to be taken first. This matters
because PaySim's fraud is not spread evenly across all 743 time steps.

Usage: python3 src/sample_real_data.py /path/to/full/AIML_Dataset.csv
Output: data/real/paysim_quarter_sample.csv (1,590,656 rows)
"""
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "data" / "real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FRACTION = 0.25


def main(source_path):
    df = pd.read_csv(source_path)
    print(f"[sample] loaded full source: {len(df):,} rows, "
          f"{df['isFraud'].sum():,} fraud ({df['isFraud'].mean():.4%})")

    sample = (
        df.groupby("isFraud", group_keys=False)
        .apply(lambda g: g.sample(frac=FRACTION, random_state=SEED))
        .sample(frac=1, random_state=SEED)  # shuffle back together
        .reset_index(drop=True)
    )

    print(f"[sample] drew stratified {FRACTION:.0%} sample: {len(sample):,} rows, "
          f"{sample['isFraud'].sum():,} fraud ({sample['isFraud'].mean():.4%})")

    out_path = OUT_DIR / "paysim_quarter_sample.csv"
    sample.to_csv(out_path, index=False)
    print(f"[sample] wrote {out_path}")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/AIML_Dataset.csv"
    main(source)
