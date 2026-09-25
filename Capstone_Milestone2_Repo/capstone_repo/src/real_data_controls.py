"""
Real-data pipeline, Part 3: security-control evaluation, mirroring the
methodology used on synthetic data (src/simulation_controls.py) so the two
are directly comparable in the report addendum.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "real"


def evaluate_control(df, held_mask, name):
    positives = df["isFraud"] == 1
    negatives = ~positives
    n_pos, n_neg = positives.sum(), negatives.sum()
    recall = float((held_mask & positives).sum() / n_pos) if n_pos else 0.0
    friction = float((held_mask & negatives).sum() / n_neg) if n_neg else 0.0
    amt = float(df.loc[held_mask & positives, "amount"].sum())
    total = float(df.loc[positives, "amount"].sum())
    return {
        "control": name, "n_held": int(held_mask.sum()),
        "recall_true_positives_held": round(recall, 4),
        "friction_false_positive_rate": round(friction, 5),
        "amount_at_risk_prevented": round(amt, 2),
        "share_of_total_fraud_amount_prevented": round(amt / total, 4) if total else 0.0,
    }


def main():
    df = pd.read_csv(OUT / "real_case_risk_table.csv")

    baseline_mask = pd.Series(False, index=df.index)
    baseline = evaluate_control(df, baseline_mask, "Baseline (no control)")

    control_a_mask = df["combined_risk"] >= 0.75
    control_a = evaluate_control(df, control_a_mask, "Control A: ML-threshold hold (t=0.75)")

    control_b_mask = df["type"].isin(["TRANSFER", "CASH_OUT"]) & (df["amount"] >= df["amount"].quantile(0.99))
    control_b = evaluate_control(df, control_b_mask, "Control B: rule-based (TRANSFER/CASH_OUT + top 1% amount)")

    combined_mask = control_a_mask | control_b_mask
    control_combined = evaluate_control(df, combined_mask, "Combined control (A OR B)")

    results = {"baseline": baseline, "control_a": control_a, "control_b": control_b,
               "control_combined": control_combined}

    sweep = []
    for t in np.arange(0.3, 0.95, 0.05):
        mask = df["combined_risk"] >= t
        r = evaluate_control(df, mask, f"t={t:.2f}")
        sweep.append({"threshold": round(float(t), 2), "recall": r["recall_true_positives_held"],
                       "friction": r["friction_false_positive_rate"], "n_held": r["n_held"]})
    results["threshold_sensitivity_sweep"] = sweep

    with open(OUT / "real_control_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[real-controls] Baseline:", baseline)
    print("[real-controls] Control A:", control_a)
    print("[real-controls] Control B:", control_b)
    print("[real-controls] Combined:", control_combined)

    thresholds = [s["threshold"] for s in sweep]
    recalls = [s["recall"] for s in sweep]
    frictions = [s["friction"] for s in sweep]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(thresholds, recalls, marker="o", color="#2E5395", label="Recall (fraud caught)")
    ax.plot(thresholds, frictions, marker="o", color="#C0504D", label="Friction (legit held)")
    ax.axvline(0.75, color="gray", linestyle="--", linewidth=1, label="Control A operating point")
    ax.set_xlabel("Combined-risk hold threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Real-data: Control A threshold sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "chart_real_control_sensitivity.png", dpi=130)
    print("[real-controls] saved chart_real_control_sensitivity.png")


if __name__ == "__main__":
    main()
