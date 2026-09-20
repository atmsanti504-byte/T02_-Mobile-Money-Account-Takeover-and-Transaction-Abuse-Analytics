import pandas as pd
import numpy as np
from pathlib import Path


# ------ FEATURE ENGINEERING
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCENARIO_DIR = DATA_DIR / "scenario"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

transactions = pd.read_csv(SCENARIO_DIR / "transactions_scenario.csv")
authentication = pd.read_csv(SCENARIO_DIR / "authentication_scenario.csv")
device_sim = pd.read_csv(SCENARIO_DIR / "device_sim_scenario.csv")
geolocation = pd.read_csv(SCENARIO_DIR / "geolocation_scenario.csv")
accounts = pd.read_csv(PROCESSED_DIR / "accounts_clean.csv")

transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
authentication["timestamp"] = pd.to_datetime(authentication["timestamp"])
device_sim["timestamp"] = pd.to_datetime(device_sim["timestamp"])
geolocation["timestamp"] = pd.to_datetime(geolocation["timestamp"])

print(f"\nTransactions: {len(transactions):,}")
print(f"Authentication events: {len(authentication):,}")
print(f"Device/SIM events: {len(device_sim):,}")
print(f"Geolocation events: {len(geolocation):,}")


#  START FEATURE TABLE

features = transactions.copy()
# TRANSACTION FEATURES
features["transaction_hour"] = (features["timestamp"].dt.hour)
features["transaction_day"] = (features["timestamp"].dt.dayofweek)
features["is_weekend"] = (features["transaction_day"] >= 5).astype(int)

#  ACCOUNT FEATURES
account_columns = [
    "account_id",
    "home_city",
    "account_age_days",
    "customer_segment",
    "preferred_channel",
    "device_platform",
    "typical_transaction_amount",
    "typical_daily_transactions",
    "baseline_risk_tier"
]

features = features.merge(
    accounts[account_columns],
    on="account_id",
    how="left"
)

# Amount relative to customer's normal transaction amount
features["amount_deviation_ratio"] = (
    features["amount"]
    / features["typical_transaction_amount"]
)

# Channel deviation
features["channel_deviation"] = (
    features["channel"]
    != features["preferred_channel"]
).astype(int)


# TRANSACTION VELOCITY
# Number of transactions per account per day
transactions["transaction_date"] = (transactions["timestamp"].dt.date)

daily_counts = (
    transactions
    .groupby(["account_id", "transaction_date"])
    .size()
    .reset_index(name="daily_transaction_count")
)

features["transaction_date"] = (features["timestamp"].dt.date)

features = features.merge(
    daily_counts,
    on=["account_id", "transaction_date"],
    how="left"
)
features = features.drop(columns=["transaction_date"])


# ACCOUNT-LEVEL AUTHENTICATION FEATURES


auth_summary = (
    authentication
    .groupby("account_id")
    .agg(
        total_login_attempts=(
            "event_id",
            "count"
        ),
        failed_login_count=(
            "login_result",
            lambda x: (x == "FAILED").sum()
        ),
        successful_login_count=(
            "login_result",
            lambda x: (x == "SUCCESS").sum()
        )).reset_index())

features = features.merge(
    auth_summary,
    on="account_id",
    how="left"
)

# Failed-login ratio
features["failed_login_ratio"] = (
    features["failed_login_count"]
    / features["total_login_attempts"]
)


# RECENT AUTHENTICATION ACTIVITY
# For each account, calculate the number of failed logins
# on the same calendar day as each transaction.
authentication["auth_date"] = (authentication["timestamp"].dt.date)

failed_daily = (
    authentication[authentication["login_result"] == "FAILED"]
    .groupby(["account_id", "auth_date"])
    .size()
    .reset_index(name="failed_logins_day")
)

features["transaction_date"] = (features["timestamp"].dt.date)

features = features.merge(
    failed_daily,
    left_on=["account_id", "transaction_date"],
    right_on=["account_id", "auth_date"],
    how="left"
)

features["failed_logins_day"] = (features["failed_logins_day"].fillna(0))

features = features.drop(
    columns=["transaction_date", "auth_date"],
    errors="ignore"
)


#DEVICE / SIM FEATURES
device_summary = (
    device_sim
    .groupby("account_id")
    .agg(
        total_device_events=(
            "event_id",
            "count"
        ),
        sim_change_count=(
            "change_type",
            lambda x: (x == "SIM_CHANGE").sum()
        ),
        device_change_count=(
            "change_type",
            lambda x: (x == "DEVICE_CHANGE").sum()
        ),
        os_update_count=(
            "change_type",
            lambda x: (x == "OS_UPDATE").sum()
        )
    ).reset_index()
)

features = features.merge(device_summary,on="account_id",how="left")

#GEOLOCATION FEATURES
# Number of distinct cities associated with each account
location_summary = (
    geolocation
    .groupby("account_id")
    .agg(distinct_cities=("city", "nunique"))
    .reset_index()
)

features = features.merge(location_summary,on="account_id",how="left")

# Whether the account has activity from multiple cities
features["multi_city_activity"] = (
    features["distinct_cities"] > 1
).astype(int)


#DEVICE / TRANSACTION RELATIONSHIP
# Count distinct devices associated with each account
device_count = (
    device_sim
    .groupby("account_id")["device_id"]
    .nunique()
    .reset_index(name="distinct_devices")
)
features = features.merge(device_count,on="account_id",how="left")


# CLEAN NUMERICAL FEATURES
numeric_features = [
    "amount",
    "account_age_days",
    "typical_transaction_amount",
    "typical_daily_transactions",
    "amount_deviation_ratio",
    "daily_transaction_count",
    "total_login_attempts",
    "failed_login_count",
    "successful_login_count",
    "failed_login_ratio",
    "failed_logins_day",
    "total_device_events",
    "sim_change_count",
    "device_change_count",
    "os_update_count",
    "distinct_cities",
    "distinct_devices"
]

for column in numeric_features:
    if column in features.columns:
        features[column] = (
            pd.to_numeric(
                features[column],
                errors="coerce"
            ).fillna(0))


# ENCODE CATEGORICAL FEATURES
categorical_features = [
    "transaction_type",
    "channel",
    "status",
    "home_city",
    "customer_segment",
    "preferred_channel",
    "device_platform",
    "baseline_risk_tier"
]

features = pd.get_dummies(
    features,
    columns=[
        column
        for column in categorical_features
        if column in features.columns],
    dtype=int)


target = features["is_fraud"].copy()
# Remove identifiers and scenario metadata
columns_to_drop = [
    "event_id",
    "account_id",
    "timestamp",
    "device_id",
    "counterparty_id",
    "scenario_type",
    "scenario_label"
]

features = features.drop(
    columns=[
        column
        for column in columns_to_drop
        if column in features.columns
    ]
)

# Put target at the end
features["is_fraud"] = target.values

# Remove duplicate merge-generated column
features = features.drop(
    columns=["typical_daily_transactions_x"],
    errors="ignore"
)

features = features.rename(
    columns={
        "typical_daily_transactions_y":
        "typical_daily_transactions"})

# Save 
output_path = FEATURE_DIR / "ato_features.csv"
features.to_csv(output_path,index=False)
print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)
