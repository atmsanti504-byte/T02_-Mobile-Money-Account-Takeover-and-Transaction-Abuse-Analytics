import os
import random
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# for reproducability
SEED = 2115799
random.seed(SEED)
np.random.seed(SEED)
print(f"Random seed: {SEED}")

# number of events to generate
N_ACCOUNTS = 5000
N_TRANSACTIONS = 100_000
N_AUTH_EVENTS = 150_000
FRAUD_RATE = 0.05
N_FRAUD_TRANSACTIONS = int(N_TRANSACTIONS * FRAUD_RATE)
N_NORMAL_TRANSACTIONS = N_TRANSACTIONS - N_FRAUD_TRANSACTIONS
print(f"Accounts: {N_ACCOUNTS:,}")
print(f"Transactions: {N_TRANSACTIONS:,}")
print(f"Expected normal transactions: {N_NORMAL_TRANSACTIONS:,}")
print(f"Expected fraudulent transactions: {N_FRAUD_TRANSACTIONS:,}")

BASE_DIR = Path("__file__").resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HASH_DIR = DATA_DIR / "hashes"

for directory in [RAW_DIR, PROCESSED_DIR, HASH_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"Project directories created at: {DATA_DIR.resolve()}.")


# -------------------------- ACCOUNT GENERATION --------------------------
# Define locations and their weights for sampling
CITIES = ["Windhoek","Swakopmund","Walvis Bay","Oshakati","Rundu","Keetmanshoop"]
CITY_WEIGHTS = [0.40,0.15,0.15,0.12,0.10,0.08]  # Sum should be 1.0

# Gen accounts
accounts = pd.DataFrame({
    "account_id": [f"NDM_{i:06d}" for i in range(1, N_ACCOUNTS + 1)],
    "home_city": np.random.choice(CITIES, size=N_ACCOUNTS, p=CITY_WEIGHTS),
    "account_age_days": np.random.randint(30, 3650, size=N_ACCOUNTS),  # Account age between 1 month and 10 years
    "customer_segment": np.random.choice(["Standard", "Premium", "Business"], size=N_ACCOUNTS, p=[0.75, 0.20, 0.05]),  # Customer segments
    "preferred_channel": np.random.choice(["USSD", "Web", "Mobile_App"], size=N_ACCOUNTS, p=[0.65, 0.25, 0.10]),  # Preferred channel
    "device_platform": np.random.choice(["Android", "iOS"], size=N_ACCOUNTS, p=[0.70, 0.30]),  # Device platform
})

# behavioral baseline 
accounts["typical_transaction_amount"] = np.round(
    np.random.lognormal(mean=np.log(500), sigma=0.8, size=N_ACCOUNTS), 2 )
# Typical daily transactions
accounts["typical_daily_transactions"] = np.random.poisson(lam=3, size=N_ACCOUNTS) + 1  
# synthetic customer risk tier 
accounts["baseline_risk_tier"] = np.random.choice(["Low", "Medium", "High"], 
                                                  size=N_ACCOUNTS, p=[0.75, 0.20, 0.05])
accounts["primary_device_id"] = [f"DEV_{i:07d}" for i in range(1, N_ACCOUNTS + 1)]
accounts["primary_sim_hash"] = [hashlib.sha256(f"SIM_{SEED}_{i}".encode()).hexdigest()[:16] for i in range(1, N_ACCOUNTS + 1)]

accounts.to_csv(RAW_DIR / "accounts.csv", index=False)
print(f"Saved {len(accounts):,} accounts.")

# -------------------------- AUTH LOG GENERATION --------------------------
START_DATE = pd.Timestamp("2026-09-01")
END_DATE = pd.Timestamp("2026-09-30 23:59:59")

auth_account_ids = np.random.choice(accounts["account_id"], size=N_AUTH_EVENTS, replace=True)
auth_timestamps = pd.to_datetime(np.random.randint(START_DATE.value // 10**9, END_DATE.value // 10**9, size=N_AUTH_EVENTS), unit='s')
# raw auth events
auth = pd.DataFrame({
    "event_id": [f"AUTH_{i:07d}" for i in range(1, N_AUTH_EVENTS + 1)],
    "timestamp": auth_timestamps,
    "account_id": auth_account_ids,
})
auth = auth.merge(accounts[["account_id", "home_city", "primary_device_id"]], 
                  on="account_id", how="left")
auth = auth.rename(columns={
    "home_city" : "ip_city",
    "primary_device_id" : "device_id"
})

# login outcomes
auth["login_result"] = np.random.choice(["SUCCESS", "FAILED"], size=len(auth), p=[0.94, 0.06])

# mfa usgae
auth["mfa_used"] = np.where(
    auth["login_result"] == "SUCCESS",
    np.random.choice([True, False], size=len(auth), p=[0.85, 0.15]),
    False
)

# auth fail reason
failure_reasons = ["Wrong Password", "Bad OTP", "Account Locked", "Expired Session"]
auth["failure_reason"] = np.where(
    auth["login_result"] == "FAILED",
    np.random.choice(failure_reasons, size=len(auth)),
    None
)

# auth temporal attributes
auth["login_hour"] = auth["timestamp"].dt.hour
auth["login_day"] = auth["timestamp"].dt.day_name()
auth["is_weekend"] = auth["timestamp"].dt.dayofweek >= 5

auth = auth.sort_values(
    ["account_id", "timestamp"]
).reset_index(drop=True)

auth.to_csv(RAW_DIR / "authentication.csv", index=False)
print(f"Saved {len(auth):,} authentication events."
      f"{RAW_DIR / 'authentication.csv'}")

# -------------------------- TRANSACTION LOG GENERATION --------------------------
TRANSACTION_TYPES = [
    "P2P_TRANSFER",
    "MERCHANT_PAYMENT",
    "BILL_PAYMENT",
    "CASH_OUT",
    "AIRTIME_PURCHASE",
    "BANK_TRANSFER"]

TRANSACTION_TYPES_WEIGHTS = [0.30, 0.30, 0.15, 0.10, 0.10, 0.05]  # Sum should be 1.0
TRANSACTION_CHANNELS = ["USSD", "Web", "Mobile_App"]
TRANSACTION_CHANNELS_WEIGHTS = [0.60, 0.25, 0.15]  # Sum should be 1.0

transaction_account_ids = np.random.choice(accounts["account_id"], 
                                           size=N_TRANSACTIONS, replace=True)
transaction_ids = [f"TXN_{i:07d}" for i in range(1, N_TRANSACTIONS + 1)]
transaction_types = np.random.choice(TRANSACTION_TYPES, size=N_TRANSACTIONS, 
                                     p=TRANSACTION_TYPES_WEIGHTS)
transaction_channels = np.random.choice(TRANSACTION_CHANNELS, size=N_TRANSACTIONS, 
                                        p=TRANSACTION_CHANNELS_WEIGHTS)
transaction_timestamps = pd.to_datetime(np.random.randint(START_DATE.value // 10**9, END_DATE.value // 10**9, size=N_TRANSACTIONS), unit='s')

#dataset creation 
transactions = pd.DataFrame({
    "transaction_id": transaction_ids,
    "timestamp": transaction_timestamps,
    "account_id": transaction_account_ids,
    "transaction_type": transaction_types,
    "channel": transaction_channels
})
print(f"Generated {len(transactions):,} transactions.")

# behavioral attributes
transactions = transactions.merge(
    accounts[
        ["account_id", "typical_transaction_amount", "typical_daily_transactions"]],
    on="account_id", how="left" )
transactions = transactions.rename(columns={"primary_device_id": "device_id"})

# transaction amount generation
transactions["amount"] = np.round(
    transactions["typical_transaction_amount"] * 
    np.random.lognormal(mean=0, sigma=0.65, size=len(transactions)), 2)
transactions = transactions.drop(
    columns=["typical_transaction_amount"]
)

# generate counterparties
N_COUNTERPARTIES = 10_000
counterparty_ids = [f"CP_{i:06d}" for i in range(1, N_COUNTERPARTIES + 1)]
transactions["counterparty_id"] = np.random.choice(
    counterparty_ids,
    size=N_TRANSACTIONS,
    replace=True
)
transactions["status"] = np.random.choice(["COMPLETED", "FAILED", "REVERSED"],
                                          size=len(transactions),p=[0.96, 0.03, 0.01])
transactions = transactions.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

transactions.to_csv(RAW_DIR / "transactions.csv",index=False)
print(f"Saved {len(transactions):,} transactions to "f"{RAW_DIR / 'transactions.csv'}")

# -------------------------- DEVICE EVENTS GENERATION --------------------------
DEVICE_MODELS = [
    "Samsung_A15",
    "Samsung_A25",
    "iPhone_13",
    "iPhone_14",
    "Huawei_Nova",
    "Xiaomi_Redmi",
    "Google_Pixel"]

OS_VERSIONS = [
    "Android_13",
    "Android_14",
    "Android_15",
    "iOS_17",
    "iOS_18"]

CHANGE_TYPES = ["SIM_CHANGE", "DEVICE_CHANGE", "OS_UPDATE"]
CHANGE_TYPES_WEIGHTS = [0.65, 0.20, 0.15]  # Sum should be 1.0

# Generate device events
N_DEVICE_EVENTS = 20_000
device_account_ids = np.random.choice(accounts["account_id"], 
                                      size=N_DEVICE_EVENTS, replace=True)
devices_timestamps = pd.to_datetime(np.random.randint(START_DATE.value // 10**9, END_DATE.value // 10**9, size=N_DEVICE_EVENTS), unit='s')
device_sim = pd.DataFrame({"event_id": [f"DEV_EVT_{i:07d}" for i in range(1, N_DEVICE_EVENTS + 1)],
                           "timestamp": devices_timestamps,
                           "account_id": device_account_ids})
device_sim["change_type"] = np.random.choice(CHANGE_TYPES,size=N_DEVICE_EVENTS,p=CHANGE_TYPES_WEIGHTS)
device_sim["device_id"] = [f"DEV_EVT_{i:07d}" for i in range(1, N_DEVICE_EVENTS + 1)]
device_sim["sim_hash"] = [hashlib.sha256(f"SIM_EVT_{SEED}_{i}".encode()).hexdigest()[:16] for i in range(1, N_DEVICE_EVENTS + 1)]
device_sim["device_model"] = np.random.choice(DEVICE_MODELS, size=N_DEVICE_EVENTS)
device_sim["os_version"] = np.random.choice(OS_VERSIONS, size=N_DEVICE_EVENTS)
device_sim = device_sim.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

device_sim.to_csv(RAW_DIR / "device_sim.csv",index=False)
print(f"Saved {len(device_sim):,} device/SIM events to "f"{RAW_DIR / 'device_sim.csv'}")

print("Shape:", device_sim.shape)

print("\nChange types:")
print(device_sim["change_type"].value_counts())

print("\nDuplicate event IDs:")
print(device_sim["event_id"].duplicated().sum())

print("\nUnknown accounts:")
print(
    (~device_sim["account_id"].isin(accounts["account_id"])).sum()
)

print("\nMissing values:")
print(device_sim.isnull().sum())

# -------------------------- GEOLOCATION EVENTS GENERATION --------------------------
GEO_CITIES = ["Windhoek","Swakopmund","Walvis Bay","Oshakati","Rundu","Keetmanshoop"]
GEO_WEIGHTS = [0.40,0.15,0.15,0.12,0.10,0.08]

# Generate geolocation events
N_GEO_EVENTS = 100_000
geo_account_ids = np.random.choice(accounts["account_id"],size=N_GEO_EVENTS,replace=True)
geo_timestamps = pd.to_datetime(np.random.randint(START_DATE.value // 10**9,END_DATE.value // 10**9,size=N_GEO_EVENTS),unit="s")
geolocation = pd.DataFrame({"event_id": [f"GEO_{i:07d}"for i in range(1, N_GEO_EVENTS + 1)],
                            "timestamp": geo_timestamps,"account_id": geo_account_ids,"city": 
                            np.random.choice(GEO_CITIES,size=N_GEO_EVENTS,p=GEO_WEIGHTS)})
geolocation = geolocation.merge(accounts[["account_id", "primary_device_id"]],on="account_id",how="left")
geolocation = geolocation.rename(columns={"primary_device_id": "device_id"})
geolocation = geolocation.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
geolocation.to_csv(RAW_DIR / "geolocation.csv",index=False)
print(f"Saved {len(geolocation):,} geolocation events to "f"{RAW_DIR / 'geolocation.csv'}")

print("Shape:", geolocation.shape)

print("\nCities:")
print(geolocation["city"].value_counts())

print("\nDuplicate event IDs:")
print(geolocation["event_id"].duplicated().sum())

print("\nUnknown accounts:")
print(
    (~geolocation["account_id"].isin(accounts["account_id"])).sum()
)

print("\nMissing values:")
print(geolocation.isnull().sum())

# -------------------------- Support Tickets --------------------------
N_SUPPORT_TICKETS = 2_000
support_account_ids = np.random.choice(accounts["account_id"],size=N_SUPPORT_TICKETS,replace=True)
support_timestamps = pd.to_datetime(np.random.randint(START_DATE.value // 10**9,END_DATE.value // 10**9,size=N_SUPPORT_TICKETS),unit="s")
support_templates = [
    "I cannot access my account and my password no longer works.",
    "There was a transaction on my account that I did not make.",
    "I received an OTP that I did not request.",
    "My phone was lost and I need to secure my account.",
    "I changed my phone and need help accessing my account.",
    "I noticed a transfer that I do not recognise.",
    "My SIM card was replaced and I cannot log in.",
    "My account is locked after several login attempts.",
    "I made this transaction myself but it is showing as pending.",
    "Please help me update my registered mobile number.",
    "I am travelling and cannot access my normal device.",
    "There is no problem with my account; I need assistance with a payment."
]
support = pd.DataFrame({"ticket_id": [f"TICKET_{i:06d}"for i in range(1, N_SUPPORT_TICKETS + 1)],
                        "timestamp": support_timestamps,"account_id": support_account_ids,"text": 
                        np.random.choice(support_templates,size=N_SUPPORT_TICKETS)})
support["priority"] = np.random.choice(["Low", "Medium", "High"],size=N_SUPPORT_TICKETS,p=[0.60, 0.30, 0.10])
support = support.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
support.to_csv(RAW_DIR / "support_tickets.csv",index=False)
print(f"Saved {len(support):,} support tickets to "f"{RAW_DIR / 'support_tickets.csv'}")















