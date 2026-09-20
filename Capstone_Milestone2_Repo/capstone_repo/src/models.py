"""
Supervised account-takeover classifier + unsupervised anomaly detector.

Both operate on the account-level feature table produced by eda_features.py.
The ground-truth label (label_account_takeover) is used ONLY for training
and evaluation of the supervised model, and for evaluating how well the
unsupervised anomaly scores align with known compromise.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, roc_auc_score,
)

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"

FEATURE_COLS = [
    "n_logins", "n_failed", "failure_rate", "mean_mfa", "geo_dispersion",
    "n_txns", "total_amount", "mean_amount", "max_amount", "n_cashout",
    "n_device_events", "n_sim_changes",
]


def run_supervised():
    df = pd.read_csv(OUT / "account_features.csv", index_col=0)
    X = df[FEATURE_COLS].fillna(0)
    y = df["label_account_takeover"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "precision": round(precision_score(y_test, preds, zero_division=0), 3),
        "recall": round(recall_score(y_test, preds, zero_division=0), 3),
        "f1": round(f1_score(y_test, preds, zero_division=0), 3),
        "pr_auc": round(average_precision_score(y_test, proba), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3) if y_test.nunique() > 1 else None,
        "n_test": int(len(y_test)),
        "n_positive_test": int(y_test.sum()),
    }
    cm = confusion_matrix(y_test, preds).tolist()

    importances = sorted(
        zip(FEATURE_COLS, clf.feature_importances_), key=lambda t: -t[1]
    )
    top_features = [{"feature": f, "importance": round(float(v), 3)} for f, v in importances[:6]]

    result = {"metrics": metrics, "confusion_matrix": cm, "top_features": top_features}
    with open(OUT / "supervised_model_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("[supervised] metrics:", metrics)
    print("[supervised] top features:", top_features[:3])

    # score full population for downstream prototype
    full_proba = clf.predict_proba(X)[:, 1]
    df["supervised_risk_score"] = full_proba
    return df, result


def run_anomaly(df, top_pct=0.05):
    X = df[FEATURE_COLS].fillna(0)
    iso = IsolationForest(n_estimators=300, contamination=top_pct, random_state=42)
    iso.fit(X)
    # more negative = more anomalous; invert & rescale to 0-1 for readability
    raw = -iso.score_samples(X)
    score = (raw - raw.min()) / (raw.max() - raw.min())
    df["anomaly_score"] = score

    threshold = np.quantile(score, 1 - top_pct)
    df["anomaly_flag"] = (score >= threshold).astype(int)

    flagged = df[df["anomaly_flag"] == 1]
    precision_at_topN = flagged["label_account_takeover"].mean() if len(flagged) else 0.0

    result = {
        "top_pct_flagged": top_pct,
        "n_flagged": int(df["anomaly_flag"].sum()),
        "precision_at_topN": round(float(precision_at_topN), 3),
        "baseline_positive_rate": round(float(df["label_account_takeover"].mean()), 3),
    }
    with open(OUT / "anomaly_model_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("[anomaly] result:", result)
    return df, result


def main():
    df, sup_result = run_supervised()
    df, anom_result = run_anomaly(df, top_pct=0.05)

    # combined case-level risk table for the prototype dashboard
    combined = df[[
        "supervised_risk_score", "anomaly_score", "anomaly_flag",
        "n_sim_changes", "failure_rate", "max_amount", "label_account_takeover",
    ]].copy()
    combined["combined_risk"] = (
        0.6 * combined["supervised_risk_score"] + 0.4 * combined["anomaly_score"]
    )
    combined = combined.sort_values("combined_risk", ascending=False)
    combined.to_csv(OUT / "case_risk_table.csv")
    print(f"[combined] wrote case_risk_table.csv with {len(combined)} accounts, "
          f"top score {combined['combined_risk'].max():.3f}")


if __name__ == "__main__":
    main()
