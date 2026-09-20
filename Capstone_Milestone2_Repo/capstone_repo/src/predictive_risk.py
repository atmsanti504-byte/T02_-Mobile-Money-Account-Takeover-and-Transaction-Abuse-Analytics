"""
Predictive risk / early-warning method + adversarial/robustness test-case plan.

Early-warning approach: rather than scoring only a completed session,
compute a rolling risk trend for a sample of accounts as their events
accumulate through the observation window, to see how early the
pipeline could have raised an alert relative to the final fraud event.

Adversarial/robustness test cases are defined (not all executed against
a production system, since none exists yet) as planned experiments for
the next milestone.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs"
FEATURE_COLS = [
    "n_logins", "n_failed", "failure_rate", "mean_mfa", "geo_dispersion",
    "n_txns", "total_amount", "mean_amount", "max_amount", "n_cashout",
    "n_device_events", "n_sim_changes",
]

ADVERSARIAL_TEST_CASES = [
    {
        "id": "ADV-1",
        "name": "Low-and-slow SIM swap",
        "description": "Attacker spaces out SIM change, login and cash-out over several days "
                        "instead of minutes, to avoid velocity-based features.",
        "status": "planned",
        "expected_risk": "Detection recall likely drops; mitigation is a longer rolling-window feature.",
    },
    {
        "id": "ADV-2",
        "name": "MFA-present takeover",
        "description": "Attacker completes MFA (e.g., via SIM-swap-enabled OTP interception), "
                        "removing the low-MFA signal the current model relies on heavily.",
        "status": "planned",
        "expected_risk": "Supervised model may under-score; anomaly model (geo/device) should partially compensate.",
    },
    {
        "id": "ADV-3",
        "name": "Small-amount test transactions",
        "description": "Attacker makes several small transactions to test account access before "
                        "a large cash-out, to stay under amount-based thresholds.",
        "status": "planned",
        "expected_risk": "Amount-based features alone insufficient; needs transaction-sequence features.",
    },
    {
        "id": "ADV-4",
        "name": "Mimicked legitimate travel",
        "description": "Attacker spoofs a gradual, realistic geolocation drift to resemble the "
                        "legitimate-travel negative-control scenario (Section: Simulation).",
        "status": "planned",
        "expected_risk": "Directly probes the false-positive/false-negative boundary found in the "
                          "legit_travel simulation; priority test for next milestone.",
    },
    {
        "id": "ADV-5",
        "name": "Feature-masking via device spoofing",
        "description": "Attacker uses a device fingerprint resembling the victim's usual device, "
                        "suppressing the device/SIM-change signal.",
        "status": "planned",
        "expected_risk": "Requires device-integrity signals beyond what current synthetic data models.",
    },
]


def rolling_early_warning(sample_accounts=6):
    auth = pd.read_csv(DATA / "auth_events.csv", parse_dates=["timestamp"])
    txns = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
    dev = pd.read_csv(DATA / "device_events.csv", parse_dates=["timestamp"])
    df = pd.read_csv(OUT / "account_features.csv", index_col=0)

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    clf.fit(df[FEATURE_COLS].fillna(0), df["label_account_takeover"])

    positive_accounts = df[df["label_account_takeover"] == 1].index.tolist()
    sample = positive_accounts[:sample_accounts]

    trends = {}
    for acc in sample:
        acc_auth = auth[auth.account_id == acc].sort_values("timestamp")
        acc_txn = txns[txns.account_id == acc].sort_values("timestamp")
        acc_dev = dev[dev.account_id == acc].sort_values("timestamp")
        if acc_auth.empty:
            continue

        checkpoints = pd.date_range(acc_auth.timestamp.min(), acc_auth.timestamp.max(), periods=5)
        series = []
        for cp in checkpoints:
            a = acc_auth[acc_auth.timestamp <= cp]
            t = acc_txn[acc_txn.timestamp <= cp]
            d = acc_dev[acc_dev.timestamp <= cp]
            n_logins = max(len(a), 1)
            feat = pd.DataFrame([{
                "n_logins": len(a), "n_failed": (a.login_result == "failed").sum(),
                "failure_rate": (a.login_result == "failed").sum() / n_logins,
                "mean_mfa": a.mfa_flag.mean() if len(a) else 0,
                "geo_dispersion": (a.latitude.std() or 0) + (a.longitude.std() or 0),
                "n_txns": len(t), "total_amount": t.amount.sum(), "mean_amount": t.amount.mean() if len(t) else 0,
                "max_amount": t.amount.max() if len(t) else 0, "n_cashout": (t.channel == "cash_out").sum(),
                "n_device_events": len(d), "n_sim_changes": (d.change_type == "sim_change").sum(),
            }]).fillna(0)[FEATURE_COLS]
            score = clf.predict_proba(feat)[0, 1]
            series.append(round(float(score), 3))
        trends[acc] = series

    with open(OUT / "early_warning_trends.json", "w") as f:
        json.dump(trends, f, indent=2)
    print("[predictive] early-warning score trends (5 checkpoints per account):")
    for acc, series in trends.items():
        print(f"   {acc}: {series}")
    return trends


def main():
    trends = rolling_early_warning()
    with open(OUT / "adversarial_test_plan.json", "w") as f:
        json.dump(ADVERSARIAL_TEST_CASES, f, indent=2)
    print(f"[predictive] wrote {len(ADVERSARIAL_TEST_CASES)} adversarial/robustness test cases (planned).")


if __name__ == "__main__":
    main()
