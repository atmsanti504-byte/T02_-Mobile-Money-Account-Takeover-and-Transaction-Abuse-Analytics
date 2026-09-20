"""
Lightweight test cases for the Milestone 2 prototype pipeline.
Run with: pytest tests/ -v   (or: python -m pytest tests/)
"""
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
DATA = BASE / "data"
OUT = BASE / "outputs"


def test_data_files_exist():
    for f in ["auth_events.csv", "transactions.csv", "device_events.csv", "support_tickets.csv"]:
        assert (DATA / f).exists(), f"missing generated data file: {f}"


def test_no_duplicate_event_ids():
    auth = pd.read_csv(DATA / "auth_events.csv")
    assert auth["event_id"].is_unique, "duplicate event_id found in auth_events.csv"


def test_transaction_amounts_positive():
    txns = pd.read_csv(DATA / "transactions.csv")
    assert (txns["amount"] > 0).all(), "found non-positive transaction amounts"


def test_account_features_built():
    feats = pd.read_csv(OUT / "account_features.csv", index_col=0)
    assert feats.shape[0] > 0
    assert "label_account_takeover" in feats.columns
    assert feats["label_account_takeover"].isin([0, 1]).all()


def test_supervised_model_beats_baseline():
    with open(OUT / "supervised_model_results.json") as f:
        result = json.load(f)
    baseline_precision = result["metrics"]["n_positive_test"] / result["metrics"]["n_test"]
    assert result["metrics"]["precision"] >= baseline_precision, (
        "supervised model precision does not beat the naive base rate"
    )
    assert result["metrics"]["pr_auc"] > 0.5, "PR-AUC should exceed random baseline of ~base rate"


def test_anomaly_flags_reasonable_share():
    with open(OUT / "anomaly_model_results.json") as f:
        result = json.load(f)
    assert 0 < result["n_flagged"] < 200, "anomaly flag count outside sane prototype range"


def test_text_classifier_above_random():
    with open(OUT / "text_mining_results.json") as f:
        result = json.load(f)
    # 4 categories -> random guessing macro F1 would be roughly 0.25
    assert result["macro_f1"] > 0.25, "text classifier macro F1 not meaningfully above random"


def test_simulation_negative_control_lower_than_attack_scenarios():
    with open(OUT / "simulation_results.json") as f:
        sim = json.load(f)
    legit = sim["legit_travel"]["mean_combined_risk"]
    attack = sim["sim_swap_wave"]["mean_combined_risk"]
    # KNOWN ISSUE (see backlog): this currently passes only on mean risk, not on
    # alert rate, since the legit-travel false-positive rate is still elevated.
    assert legit < attack, "negative control scenario should score lower than the true attack scenario"


def test_case_risk_table_sorted_descending():
    df = pd.read_csv(OUT / "case_risk_table.csv", index_col=0)
    assert df["combined_risk"].is_monotonic_decreasing, "case risk table is not sorted by risk descending"
