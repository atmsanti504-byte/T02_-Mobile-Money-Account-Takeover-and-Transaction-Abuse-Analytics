"""
Simulation design: three scenarios injected into the synthetic environment,
scored with the trained supervised model + anomaly detector, to compare
how the pipeline responds to different threat/behaviour patterns.

Scenarios:
  1. SIM-swap wave        - burst of SIM changes followed by cash-out
  2. Credential stuffing  - many failed logins across accounts, low value txns
  3. Legitimate travel spike - genuine customers travelling (geo shift, no fraud)

The third scenario is a NEGATIVE control: a good pipeline should raise
few/no alerts for it, since it represents legitimate unusual behaviour.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
FEATURE_COLS = [
    "n_logins", "n_failed", "failure_rate", "mean_mfa", "geo_dispersion",
    "n_txns", "total_amount", "mean_amount", "max_amount", "n_cashout",
    "n_device_events", "n_sim_changes",
]

rng = np.random.default_rng(7)


def base_features(df):
    return df[FEATURE_COLS].fillna(0)


def make_scenario_accounts(n, kind):
    """Generate synthetic account-level feature rows for a scenario."""
    rows = []
    for i in range(n):
        if kind == "sim_swap_wave":
            row = dict(
                n_logins=rng.integers(3, 8), n_failed=rng.integers(1, 4),
                mean_mfa=rng.uniform(0.0, 0.3), geo_dispersion=rng.uniform(2, 6),
                n_txns=rng.integers(1, 3), total_amount=rng.uniform(20000, 80000),
                mean_amount=rng.uniform(15000, 60000), max_amount=rng.uniform(20000, 80000),
                n_cashout=rng.integers(1, 3), n_device_events=rng.integers(1, 2),
                n_sim_changes=1,
            )
        elif kind == "credential_stuffing":
            row = dict(
                n_logins=rng.integers(10, 25), n_failed=rng.integers(8, 22),
                mean_mfa=rng.uniform(0.0, 0.2), geo_dispersion=rng.uniform(1, 4),
                n_txns=rng.integers(0, 1), total_amount=rng.uniform(0, 500),
                mean_amount=rng.uniform(0, 500), max_amount=rng.uniform(0, 500),
                n_cashout=0, n_device_events=rng.integers(0, 2),
                n_sim_changes=0,
            )
        elif kind == "legit_travel":
            row = dict(
                n_logins=rng.integers(4, 10), n_failed=rng.integers(0, 2),
                mean_mfa=rng.uniform(0.7, 1.0), geo_dispersion=rng.uniform(1.5, 3.5),
                n_txns=rng.integers(2, 6), total_amount=rng.uniform(1000, 6000),
                mean_amount=rng.uniform(500, 2000), max_amount=rng.uniform(1000, 3000),
                n_cashout=rng.integers(0, 1), n_device_events=0,
                n_sim_changes=0,
            )
        row["failure_rate"] = row["n_failed"] / max(row["n_logins"], 1)
        rows.append(row)
    return pd.DataFrame(rows)[FEATURE_COLS]


def main():
    df = pd.read_csv(OUT / "account_features.csv", index_col=0)
    X = base_features(df)
    y = df["label_account_takeover"]

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    clf.fit(X, y)
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    iso.fit(X)
    raw = -iso.score_samples(X)
    anom_min, anom_max = raw.min(), raw.max()

    scenarios = {
        "sim_swap_wave": {"n": 25, "expected": "high alert rate (true attack pattern)"},
        "credential_stuffing": {"n": 30, "expected": "moderate-high alert rate via failure-rate/anomaly signal"},
        "legit_travel": {"n": 25, "expected": "low alert rate (negative control, legitimate behaviour)"},
    }

    results = {}
    for name, cfg in scenarios.items():
        sim_df = make_scenario_accounts(cfg["n"], name)
        proba = clf.predict_proba(sim_df)[:, 1]
        anom_raw = -iso.score_samples(sim_df)
        anom_score = (anom_raw - anom_min) / (anom_max - anom_min)
        combined = 0.6 * proba + 0.4 * anom_score

        alert_rate = float((combined >= 0.5).mean())
        results[name] = {
            "n_simulated_accounts": cfg["n"],
            "expected_pattern": cfg["expected"],
            "mean_supervised_score": round(float(proba.mean()), 3),
            "mean_anomaly_score": round(float(anom_score.mean()), 3),
            "mean_combined_risk": round(float(combined.mean()), 3),
            "alert_rate_at_0.5": round(alert_rate, 3),
        }
        print(f"[simulation:{name}] {results[name]}")

    with open(OUT / "simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[simulation] wrote simulation_results.json")


if __name__ == "__main__":
    main()
