import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

accounts = pd.read_csv(PROCESSED_DIR / "accounts_clean.csv")
authentication = pd.read_csv(PROCESSED_DIR / "authentication_clean.csv")
transactions = pd.read_csv(PROCESSED_DIR / "transactions_clean.csv")
device_sim = pd.read_csv(PROCESSED_DIR / "device_sim_clean.csv")
geolocation = pd.read_csv(PROCESSED_DIR / "geolocation_clean.csv")
support = pd.read_csv(PROCESSED_DIR / "support_tickets_clean.csv")

authentication["timestamp"] = pd.to_datetime(authentication["timestamp"])
transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
device_sim["timestamp"] = pd.to_datetime(device_sim["timestamp"])
geolocation["timestamp"] = pd.to_datetime(geolocation["timestamp"])
support["timestamp"] = pd.to_datetime(support["timestamp"])

print("=" * 60)
print("EDA AND BASELINE BEHAVIOUR")
print("=" * 60)

# ------ Authentication Baseline ------
print("\nAUTHENTICATION BASELINE")
print("\nLogin results:")
print(authentication["login_result"].value_counts())
print("\nLogin results (%):")
print(authentication["login_result"].value_counts(normalize=True).round(4) * 100)
print("\nFailed login reasons:")
print(authentication.loc[authentication["login_result"] == "FAILED","failure_reason"].value_counts())
print("\nLogins by hour:")
print(authentication["login_hour"].value_counts().sort_index())

# ------ Transactions Baseline ------
print("\nTRANSACTION BASELINE")
print("\nTransaction types:")
print(transactions["transaction_type"].value_counts())
print("\nTransaction channels:")
print(transactions["channel"].value_counts())
print("\nTransaction status:")
print(transactions["status"].value_counts())
print("\nTransaction amount statistics:")
print(transactions["amount"].describe())

# ------ Device/SIM Baseline ------
print("\nDEVICE/SIM BASELINE")
print(device_sim["change_type"].value_counts())
print("\nChanges per account:")
changes_per_account = device_sim.groupby("account_id").size()
print(changes_per_account.describe())

#------ Geolocation Baseline ------
print("\nGEOLOCATION BASELINE")
print(geolocation["city"].value_counts())
print("\nLocations per account:")
locations_per_account = geolocation.groupby("account_id")["city"].nunique()
print(locations_per_account.describe())

# ----- Account Activity Baseline ------
print("\nACCOUNT ACTIVITY BASELINE")
auth_per_account = authentication.groupby("account_id").size()
txn_per_account = transactions.groupby("account_id").size()
account_activity = accounts[["account_id"]].copy()
account_activity["authentication_events"] = (account_activity["account_id"].map(auth_per_account).fillna(0))
account_activity["transaction_events"] = (account_activity["account_id"].map(txn_per_account).fillna(0))
print(account_activity.describe())

# ---- Security visualizations -----

# --- Authentication Activity by Hour
plt.figure(figsize=(10, 5))
authentication["login_hour"].value_counts().sort_index().plot(kind="bar")
plt.title("Authentication Activity by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Authentication Events")
plt.tight_layout()
plt.savefig(DATA_DIR / "auth_activity_by_hour.png", dpi=300)
plt.show()
# --- Transaction Amount Distribution
plt.figure(figsize=(10, 5))
transactions["amount"].clip(upper=5000).plot(
    kind="hist",
    bins=50
)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(DATA_DIR / "transaction_amount_distribution.png", dpi=300)
plt.show()













