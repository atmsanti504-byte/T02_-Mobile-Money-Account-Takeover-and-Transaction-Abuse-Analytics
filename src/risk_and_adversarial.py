from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURE_PATH = DATA_DIR / "features" / "ato_features.csv"
ANOMALY_PATH = DATA_DIR / "isolation_forest_results.csv"
SUPPORT_PATH = DATA_DIR / "scenario" / "support_scenario.csv"
XGB_PATH = DATA_DIR / "xgboost_ato_model.json"
SEED = 2115799

#transact features
df = pd.read_csv(FEATURE_PATH)
transaction_ids = df["transaction_id"].copy()
true_labels = df["is_fraud"].copy()
# Clean model features
drop_columns = ["transaction_id","is_fraud","typical_daily_transactions_x"]
model_df = df.drop(columns=[c for c in drop_columns if c in df.columns])

if "typical_daily_transactions_y" in model_df.columns:
    model_df = model_df.rename(
        columns={
            "typical_daily_transactions_y":
            "typical_daily_transactions"
        }
    )
X = model_df


# load XGBOOST
print("=" * 60)
print("ATO RISK SCORING + ADVERSARIAL TESTING")
print("=" * 60)
xgb_model = XGBClassifier()
xgb_model.load_model(XGB_PATH)
xgb_probability = xgb_model.predict_proba(X)[:, 1]
print("XGBoost predictions generated.")


#load isolation froest results
anomaly = pd.read_csv(ANOMALY_PATH)
anomaly = anomaly[["transaction_id", "anomaly_score"]]

risk_df = pd.DataFrame({
    "transaction_id": transaction_ids,
    "is_fraud": true_labels,
    "xgb_probability": xgb_probability})

risk_df = risk_df.merge(anomaly,on="transaction_id",how="left")


#normalise annomaly score
# Convert anomaly score to a percentile-style 0-1 signal.
risk_df["anomaly_signal"] = (
    risk_df["anomaly_score"]
    .rank(pct=True)
)

# nlp support ticket indicator
support = pd.read_csv(SUPPORT_PATH)
TEXT_COLUMN = "text"
support[TEXT_COLUMN] = (
    support[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

support["nlp_target"] = (
    support["scenario_label"]
    .map({
        "LEGITIMATE": 0,
        "ATO": 1
    })
)

support = support.dropna(subset=["nlp_target"])

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000)

X_text = vectorizer.fit_transform(support[TEXT_COLUMN])

nlp_model = LogisticRegression(max_iter=1000,class_weight="balanced",random_state=SEED)
nlp_model.fit(X_text,support["nlp_target"])
support["nlp_probability"] = (nlp_model.predict_proba(X_text)[:, 1])

# Maximum ATO probability associated with each account.
account_nlp = (
    support
    .groupby("account_id")["nlp_probability"]
    .max()
    .reset_index()
)

account_nlp.columns = ["account_id","nlp_signal"]

# Get account IDs from the scenario transaction data.
transaction_context = pd.read_csv(DATA_DIR / "scenario" / "transactions_scenario.csv")
transaction_context = transaction_context[["transaction_id", "account_id"]].drop_duplicates()
risk_df = risk_df.merge(transaction_context,on="transaction_id",how="left")
risk_df = risk_df.merge(account_nlp,on="account_id",how="left")
risk_df["nlp_signal"] = (risk_df["nlp_signal"].fillna(0))

# ato risk score
risk_df["risk_score"] = (
    0.50 * risk_df["xgb_probability"]
    + 0.30 * risk_df["anomaly_signal"]
    + 0.20 * risk_df["nlp_signal"])

# risk cats
def assign_risk(score):

    if score >= 0.75:
        return "CRITICAL"

    elif score >= 0.50:
        return "HIGH"

    elif score >= 0.25:
        return "MEDIUM"

    return "LOW"

risk_df["risk_level"] = (risk_df["risk_score"].apply(assign_risk))

#risk summary
print("\n" + "=" * 60)
print("RISK SCORE SUMMARY")
print("=" * 60)
print(risk_df["risk_level"].value_counts().sort_index())
print("\nMean risk score by true class:")
print(risk_df.groupby("is_fraud")["risk_score"].mean())


# feature manipulation advesary test 1 
print("\n" + "=" * 60)
print("ADVERSARIAL TEST 1 - FEATURE MANIPULATION")
print("=" * 60)
manipulated = X.copy()
if "failed_login_count" in manipulated.columns:
    manipulated["failed_login_count"] = (manipulated["failed_login_count"] * 0.20)

manipulated_probability = (xgb_model.predict_proba(manipulated)[:, 1])

baseline_mean = xgb_probability.mean()
manipulated_mean = manipulated_probability.mean()
print(f"Baseline mean XGBoost probability: "f"{baseline_mean:.4f}")
print(f"Manipulated mean probability: "f"{manipulated_mean:.4f}")


# low-and-slow attack advesary test 2
print("\n" + "=" * 60)
print("ADVERSARIAL TEST 2 - LOW-AND-SLOW")
print("=" * 60)
slow = X.copy()

# Reduce obvious velocity signals.
for feature in [
    "failed_login_count",
    "failed_logins_day",
    "daily_transaction_count"
]:
    if feature in slow.columns:
        slow[feature] = (
            slow[feature] * 0.50
            )

slow_probability = (xgb_model.predict_proba(slow)[:, 1])
print(f"Baseline mean probability: "f"{xgb_probability.mean():.4f}")
print(f"Low-and-slow probability: "f"{slow_probability.mean():.4f}")

# test obfuscation advesarial test 3
print("\n" + "=" * 60)
print("ADVERSARIAL TEST 3 - TEXT OBFUSCATION")
print("=" * 60)

obfuscated_text = (
    support[TEXT_COLUMN]
    .str.replace("lost", "l0st", regex=False)
    .str.replace("password", "passw0rd", regex=False)
    .str.replace("account", "acc0unt", regex=False)
    .str.replace("transaction", "transacti0n", regex=False)
)

obfuscated_matrix = vectorizer.transform(obfuscated_text)
obfuscated_probability = (nlp_model.predict_proba(obfuscated_matrix)[:, 1])
normal_nlp_mean = support["nlp_probability"].mean()
obfuscated_nlp_mean = (obfuscated_probability.mean())
print(f"Baseline mean NLP probability: "f"{normal_nlp_mean:.4f}")
print(f"Obfuscated mean NLP probability: "f"{obfuscated_nlp_mean:.4f}")

risk_df.to_csv(DATA_DIR / "ato_risk_scores.csv",index=False)

# saving adversary results
adversarial_results = pd.DataFrame({
    "test": [
        "Feature Manipulation",
        "Low-and-Slow",
        "Text Obfuscation"
    ],
    "baseline_score": [
        baseline_mean,
        xgb_probability.mean(),
        normal_nlp_mean
    ],
    "adversarial_score": [
        manipulated_mean,
        slow_probability.mean(),
        obfuscated_nlp_mean
    ]})

adversarial_results["score_change"] = (adversarial_results["adversarial_score"]- adversarial_results["baseline_score"])
adversarial_results.to_csv(DATA_DIR / "adversarial_test_results.csv",index=False)

# top high risk cases
print("\n" + "=" * 60)
print("TOP 10 HIGH-RISK TRANSACTIONS")
print("=" * 60)

print(
    risk_df
    .sort_values("risk_score", ascending=False)
    [["transaction_id","account_id","xgb_probability",
      "anomaly_signal","nlp_signal","risk_score","risk_level"]]
    .head(10).to_string(index=False))

print("\n" + "=" * 60)
print("RISK + ADVERSARIAL TESTING COMPLETE")
print("=" * 60)
print("Saved:")
print("- data/ato_risk_scores.csv")
print("- data/adversarial_test_results.csv")