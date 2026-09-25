"""
Simulation and security-control evaluation (Milestone 2, Section 9).

Distinct from simulation.py (which injects attack/behaviour SCENARIOS),
this module evaluates CONTROLS: given the same trained pipeline and the
same population of accounts, what happens under different response
policies applied to the combined risk score?

Baseline:    no control -> nothing is held, all risk is "realised"
Control A:   ML-threshold hold -> any account with combined_risk >= t is
             held for step-up verification before cash-out
Control B:   rule-based step-up -> any account with a SIM/device change
             AND a cash-out transaction is held for step-up verification,
             regardless of ML score (defence-in-depth control)

For each control we report:
  - recall (share of TRUE compromised accounts the control would have held)
  - friction (share of LEGITIMATE accounts incorrectly held - customer cost)
  - amount at risk prevented (sum of max_amount for compromised accounts held)

A sensitivity sweep varies the ML threshold for Control A from 0.3 to 0.9.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"


def load():
    df = pd.read_csv(OUT / "case_risk_table.csv", index_col=0)
    return df


def evaluate_control(df, held_mask, name):
    positives = df["label_account_takeover"] == 1
    negatives = ~positives
    n_pos = positives.sum()
    n_neg = negatives.sum()

    recall = float((held_mask & positives).sum() / n_pos) if n_pos else 0.0
    friction = float((held_mask & negatives).sum() / n_neg) if n_neg else 0.0
    amount_prevented = float(df.loc[held_mask & positives, "max_amount"].sum())
    total_fraud_amount = float(df.loc[positives, "max_amount"].sum())

    return {
        "control": name,
        "n_held": int(held_mask.sum()),
        "recall_true_positives_held": round(recall, 3),
        "friction_false_positive_rate": round(friction, 3),
        "amount_at_risk_prevented": round(amount_prevented, 2),
        "share_of_total_fraud_amount_prevented": round(
            amount_prevented / total_fraud_amount, 3
        ) if total_fraud_amount else 0.0,
    }


def main():
    df = load()

    # Baseline: no control
    baseline_mask = pd.Series(False, index=df.index)
    baseline = evaluate_control(df, baseline_mask, "Baseline (no control)")

    # Control A: ML-threshold hold at t=0.75 (matches dashboard 'High' tier)
    control_a_mask = df["combined_risk"] >= 0.75
    control_a = evaluate_control(df, control_a_mask, "Control A: ML-threshold hold (t=0.75)")

    # Control B: rule-based defence-in-depth (SIM change AND cash-out-scale amount)
    control_b_mask = (df["n_sim_changes"] >= 1) & (df["max_amount"] >= 10000)
    control_b = evaluate_control(df, control_b_mask, "Control B: rule-based step-up (SIM change + high-value txn)")

    # Combined control: hold if EITHER A or B would hold (defence-in-depth layering)
    combined_mask = control_a_mask | control_b_mask
    control_combined = evaluate_control(df, combined_mask, "Combined control (A OR B)")

    results = {"baseline": baseline, "control_a": control_a, "control_b": control_b,
               "control_combined": control_combined}

    # Sensitivity sweep for Control A threshold
    sweep = []
    for t in np.arange(0.3, 0.95, 0.05):
        mask = df["combined_risk"] >= t
        r = evaluate_control(df, mask, f"t={t:.2f}")
        sweep.append({
            "threshold": round(float(t), 2),
            "recall": r["recall_true_positives_held"],
            "friction": r["friction_false_positive_rate"],
            "n_held": r["n_held"],
        })
    results["threshold_sensitivity_sweep"] = sweep

    with open(OUT / "control_simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[controls] Baseline:", baseline)
    print("[controls] Control A:", control_a)
    print("[controls] Control B:", control_b)
    print(f"[controls] wrote sensitivity sweep with {len(sweep)} threshold points")

    # sensitivity chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [s["threshold"] for s in sweep]
    recalls = [s["recall"] for s in sweep]
    frictions = [s["friction"] for s in sweep]

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(thresholds, recalls, marker="o", color="#2E5395", label="Recall (fraud caught)")
    ax.plot(thresholds, frictions, marker="o", color="#C0504D", label="Friction (legit accounts held)")
    ax.axvline(0.75, color="gray", linestyle="--", linewidth=1, label="Control A operating point")
    ax.set_xlabel("Combined-risk hold threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Control A threshold sensitivity: recall vs. customer friction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "chart_control_sensitivity.png", dpi=130)
    print("[controls] saved chart_control_sensitivity.png")


if __name__ == "__main__":
    main()
