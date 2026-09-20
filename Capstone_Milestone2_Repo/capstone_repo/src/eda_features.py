"""
Exploratory analysis and feature engineering.

Joins authentication, transaction and device-change events on account_id,
computes account-level behavioural features, and saves baseline
visualisations + a data dictionary + a cleaning/quality-check log.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def load_and_clean():
    auth = pd.read_csv(DATA / "auth_events.csv", parse_dates=["timestamp"])
    txns = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
    dev = pd.read_csv(DATA / "device_events.csv", parse_dates=["timestamp"])
    tickets = pd.read_csv(DATA / "support_tickets.csv", parse_dates=["timestamp"])

    for name, df in [("auth", auth), ("transactions", txns), ("device", dev), ("tickets", tickets)]:
        n_before = len(df)
        n_dupes = df.duplicated().sum()
        n_nulls = df.isna().sum().sum()
        df.drop_duplicates(inplace=True)
        log(f"[quality-check] {name}: {n_before} rows -> {len(df)} after de-dup "
            f"({n_dupes} duplicates removed); {n_nulls} null cells found.")

    # basic range checks
    bad_amounts = (txns["amount"] <= 0).sum()
    if bad_amounts:
        txns = txns[txns["amount"] > 0]
    log(f"[quality-check] transactions: removed {bad_amounts} non-positive amount rows.")

    return auth, txns, dev, tickets


def build_account_features(auth, txns, dev):
    # authentication features
    auth_feat = auth.groupby("account_id").agg(
        n_logins=("event_id", "count"),
        n_failed=("login_result", lambda s: (s == "failed").sum()),
        mean_mfa=("mfa_flag", "mean"),
        lat_std=("latitude", "std"),
        lon_std=("longitude", "std"),
    ).fillna(0)
    auth_feat["failure_rate"] = auth_feat["n_failed"] / auth_feat["n_logins"].clip(lower=1)
    auth_feat["geo_dispersion"] = (auth_feat["lat_std"] + auth_feat["lon_std"])

    # transaction features
    txn_feat = txns.groupby("account_id").agg(
        n_txns=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        n_cashout=("channel", lambda s: (s == "cash_out").sum()),
    ).fillna(0)

    # device features
    dev_feat = dev.groupby("account_id").agg(
        n_device_events=("event_id", "count"),
        n_sim_changes=("change_type", lambda s: (s == "sim_change").sum()),
    ).fillna(0)

    # ground truth label (any takeover-flagged auth event OR fraud txn OR suspicious device event)
    label_auth = auth.groupby("account_id")["is_account_takeover"].max()
    label_txn = txns.groupby("account_id")["is_fraud"].max()
    label_dev = dev.groupby("account_id")["is_suspicious"].max()

    features = auth_feat.join(txn_feat, how="outer").join(dev_feat, how="outer").fillna(0)
    label = pd.concat([label_auth, label_txn, label_dev], axis=1).fillna(0).max(axis=1)
    features["label_account_takeover"] = label.reindex(features.index).fillna(0).astype(int)

    features.to_csv(OUT / "account_features.csv")
    log(f"[features] built account-level feature table: {features.shape[0]} accounts x {features.shape[1]} columns")
    log(f"[features] positive label rate: {features['label_account_takeover'].mean():.3f}")
    return features


def make_baseline_charts(auth, txns, features):
    # 1. login result distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    auth["login_result"].value_counts().plot(kind="bar", ax=ax, color=["#2E5395", "#C0504D"])
    ax.set_title("Login outcome distribution")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(OUT / "chart_login_outcomes.png", dpi=130)
    plt.close(fig)

    # 2. transaction amount distribution (log scale) split by fraud label
    fig, ax = plt.subplots(figsize=(5, 4))
    for lab, color, name in [(0, "#2E5395", "legitimate"), (1, "#C0504D", "fraud")]:
        subset = txns[txns["is_fraud"] == lab]["amount"]
        ax.hist(np.log1p(subset), bins=30, alpha=0.6, label=name, color=color)
    ax.set_title("Transaction amount (log) by label")
    ax.set_xlabel("log(1+amount)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "chart_txn_amount_by_label.png", dpi=130)
    plt.close(fig)

    # 3. failure rate vs geo dispersion scatter, coloured by label
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = features["label_account_takeover"].map({0: "#2E5395", 1: "#C0504D"})
    ax.scatter(features["failure_rate"], features["geo_dispersion"], c=colors, alpha=0.6, s=18)
    ax.set_xlabel("login failure rate")
    ax.set_ylabel("geolocation dispersion")
    ax.set_title("Behavioural separation: failure rate vs geo dispersion")
    fig.tight_layout()
    fig.savefig(OUT / "chart_behaviour_scatter.png", dpi=130)
    plt.close(fig)
    log("[charts] saved 3 baseline charts to outputs/")


DATA_DICTIONARY = {
    "auth_events.csv": {
        "event_id": "unique authentication event id",
        "account_id": "account identifier (synthetic)",
        "timestamp": "event timestamp",
        "device_id": "hashed device identifier",
        "login_result": "success | failed",
        "mfa_flag": "1 if multi-factor auth was used",
        "failure_reason": "reason code for failed logins",
        "latitude/longitude": "coarse, city-level synthetic location",
        "is_account_takeover": "GROUND TRUTH synthetic label, evaluation only, not a model input",
    },
    "transactions.csv": {
        "transaction_id": "unique transaction id",
        "account_id": "account identifier (synthetic)",
        "timestamp": "transaction timestamp",
        "amount": "transaction amount (synthetic currency units)",
        "channel": "p2p_transfer | merchant_payment | cash_out | bill_payment",
        "counterparty_id": "hashed counterparty identifier",
        "is_fraud": "GROUND TRUTH synthetic label, evaluation only, not a model input",
    },
    "device_events.csv": {
        "event_id": "unique device/SIM change event id",
        "account_id": "account identifier (synthetic)",
        "change_type": "sim_change | device_change | os_update",
        "device_model_hash": "hashed device model identifier",
        "is_suspicious": "GROUND TRUTH synthetic label, evaluation only, not a model input",
    },
    "support_tickets.csv": {
        "ticket_id": "unique ticket id",
        "account_id_hashed": "hashed account identifier",
        "text": "synthetic free-text complaint",
        "category": "account_takeover | billing_dispute | technical_issue | general_inquiry",
    },
    "account_features.csv": {
        "n_logins/n_failed/failure_rate": "login volume and failure behaviour per account",
        "mean_mfa": "share of logins using MFA",
        "geo_dispersion": "spread of login coordinates (proxy for location anomaly)",
        "n_txns/total_amount/mean_amount/max_amount/n_cashout": "transaction behaviour per account",
        "n_device_events/n_sim_changes": "device/SIM change activity per account",
        "label_account_takeover": "GROUND TRUTH synthetic label used only for evaluation",
    },
}


def main():
    auth, txns, dev, tickets = load_and_clean()
    features = build_account_features(auth, txns, dev)
    make_baseline_charts(auth, txns, features)

    with open(OUT / "data_dictionary.json", "w") as f:
        json.dump(DATA_DICTIONARY, f, indent=2)
    with open(OUT / "cleaning_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    log("[done] EDA and feature engineering complete.")


if __name__ == "__main__":
    main()
