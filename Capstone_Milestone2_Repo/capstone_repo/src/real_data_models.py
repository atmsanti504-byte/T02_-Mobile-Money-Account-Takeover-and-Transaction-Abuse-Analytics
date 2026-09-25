"""
Real-data pipeline, Part 2: supervised classifier + unsupervised anomaly
detector on the PaySim quarter sample (transaction-level, real labels).

Because real fraud only ever occurs within TRANSFER/CASH_OUT transactions
in this dataset (verified in real_data_eda.py), both models are trained
on the full transaction population but this is noted explicitly, since it
materially affects how precision/recall figures should be read compared
to the synthetic account-level models in Milestone 2/3.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "real"


def run_supervised():
    df = pd.read_csv(OUT / "real_transactions_engineered.csv")
    feature_cols = json.load(open(OUT / "feature_cols.json"))
    X = df[feature_cols].fillna(0)
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Single-core container: subsample the LEGITIMATE training rows to keep
    # RandomForest training tractable, while keeping every fraud example and
    # the FULL, untouched, correctly-stratified test set for honest evaluation.
    train_df = X_train.copy()
    train_df["isFraud"] = y_train.values
    fraud_train = train_df[train_df["isFraud"] == 1]
    legit_train = train_df[train_df["isFraud"] == 0].sample(n=150_000, random_state=42)
    train_sub = pd.concat([fraud_train, legit_train]).sample(frac=1, random_state=42)
    X_train_sub = train_sub[feature_cols]
    y_train_sub = train_sub["isFraud"]
    print(f"[real-supervised] training on reduced set: {len(X_train_sub)} rows "
          f"({fraud_train.shape[0]} fraud + {legit_train.shape[0]} legit); "
          f"evaluating on FULL test set: {len(X_test)} rows")

    clf = RandomForestClassifier(
        n_estimators=120, max_depth=8, class_weight="balanced", random_state=42, n_jobs=1
    )
    clf.fit(X_train_sub, y_train_sub)

    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "precision": round(precision_score(y_test, preds, zero_division=0), 3),
        "recall": round(recall_score(y_test, preds, zero_division=0), 3),
        "f1": round(f1_score(y_test, preds, zero_division=0), 3),
        "pr_auc": round(average_precision_score(y_test, proba), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "n_test": int(len(y_test)),
        "n_positive_test": int(y_test.sum()),
    }
    cm = confusion_matrix(y_test, preds).tolist()
    importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda t: -t[1])
    top_features = [{"feature": f, "importance": round(float(v), 4)} for f, v in importances[:8]]

    result = {
        "dataset": "PaySim quarter sample (real, 1,590,656 transactions, 2,054 fraud, 0.129% base rate)",
        "metrics": metrics, "confusion_matrix": cm, "top_features": top_features,
    }
    with open(OUT / "real_supervised_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("[real-supervised] metrics:", metrics)
    print("[real-supervised] top features:", top_features[:4])

    # charts
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm_arr, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_arr[i,j]:,}", ha="center", va="center",
                     color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: legit", "Pred: fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: legit", "Actual: fraud"])
    ax.set_title("Real-data confusion matrix\n(held-out test transactions)")
    fig.tight_layout()
    fig.savefig(OUT / "chart_real_confusion_matrix.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    names = [f["feature"] for f in top_features][::-1]
    vals = [f["importance"] for f in top_features][::-1]
    ax.barh(names, vals, color="#2E5395")
    ax.set_title("Real-data: top feature importances")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(OUT / "chart_real_feature_importance.png", dpi=130)
    plt.close(fig)

    df["real_supervised_score"] = clf.predict_proba(X)[:, 1]
    return df, feature_cols, result


def run_anomaly(df, feature_cols, top_pct=0.01):
    X = df[feature_cols].fillna(0)
    # subsample for IsolationForest fit speed (single-core container) on 1.59M rows; score full population after
    fit_sample = X.sample(n=min(100_000, len(X)), random_state=42)
    iso = IsolationForest(n_estimators=100, contamination=top_pct, random_state=42, n_jobs=1)
    iso.fit(fit_sample)

    raw = -iso.score_samples(X)
    score = (raw - raw.min()) / (raw.max() - raw.min())
    df["real_anomaly_score"] = score
    threshold = np.quantile(score, 1 - top_pct)
    df["real_anomaly_flag"] = (score >= threshold).astype(int)

    flagged = df[df["real_anomaly_flag"] == 1]
    precision_at_topN = flagged["isFraud"].mean() if len(flagged) else 0.0
    recall_at_topN = flagged["isFraud"].sum() / df["isFraud"].sum() if df["isFraud"].sum() else 0.0

    result = {
        "top_pct_flagged": top_pct,
        "n_flagged": int(df["real_anomaly_flag"].sum()),
        "precision_at_topN": round(float(precision_at_topN), 4),
        "recall_at_topN": round(float(recall_at_topN), 4),
        "baseline_positive_rate": round(float(df["isFraud"].mean()), 5),
    }
    with open(OUT / "real_anomaly_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("[real-anomaly] result:", result)
    return df, result


def main():
    df, feature_cols, sup_result = run_supervised()
    df, anom_result = run_anomaly(df, feature_cols, top_pct=0.01)

    combined = df[[
        "real_supervised_score", "real_anomaly_score", "real_anomaly_flag",
        "type", "amount", "isFraud",
    ]].copy()
    combined["combined_risk"] = 0.7 * combined["real_supervised_score"] + 0.3 * combined["real_anomaly_score"]
    combined = combined.sort_values("combined_risk", ascending=False)
    combined.to_csv(OUT / "real_case_risk_table.csv", index=False)
    print(f"[combined] wrote real_case_risk_table.csv, {len(combined)} rows, "
          f"top score {combined['combined_risk'].max():.3f}")


if __name__ == "__main__":
    main()
