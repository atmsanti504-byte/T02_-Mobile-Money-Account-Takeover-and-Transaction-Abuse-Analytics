import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

accounts = pd.read_csv(RAW_DIR / "accounts.csv")
authentication = pd.read_csv(RAW_DIR / "authentication.csv")
transactions = pd.read_csv(RAW_DIR / "transactions.csv")
device_sim = pd.read_csv(RAW_DIR / "device_sim.csv")
geolocation = pd.read_csv(RAW_DIR / "geolocation.csv")
support = pd.read_csv(RAW_DIR / "support_tickets.csv")

authentication["timestamp"] = pd.to_datetime(authentication["timestamp"])
transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
device_sim["timestamp"] = pd.to_datetime(device_sim["timestamp"])
geolocation["timestamp"] = pd.to_datetime(geolocation["timestamp"])
support["timestamp"] = pd.to_datetime(support["timestamp"])

authentication = authentication.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
transactions = transactions.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
device_sim = device_sim.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
geolocation = geolocation.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
support = support.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

authentication["failure_reason"] = authentication["failure_reason"].fillna("Not_Applicable")
accounts = accounts.drop_duplicates()
authentication = authentication.drop_duplicates()
transactions = transactions.drop_duplicates()
device_sim = device_sim.drop_duplicates()
geolocation = geolocation.drop_duplicates()
support = support.drop_duplicates()

accounts.to_csv(PROCESSED_DIR / "accounts_clean.csv", index=False)
authentication.to_csv(PROCESSED_DIR / "authentication_clean.csv", index=False)
transactions.to_csv(PROCESSED_DIR / "transactions_clean.csv", index=False)
device_sim.to_csv(PROCESSED_DIR / "device_sim_clean.csv", index=False)
geolocation.to_csv(PROCESSED_DIR / "geolocation_clean.csv", index=False)
support.to_csv(PROCESSED_DIR / "support_tickets_clean.csv", index=False)

print("\nProcessed datasets:")

processed_datasets = {
    "accounts": accounts,
    "authentication": authentication,
    "transactions": transactions,
    "device_sim": device_sim,
    "geolocation": geolocation,
    "support_tickets": support
}

for name, df in processed_datasets.items():
    print(f"{name}: {df.shape}")

print("\nPreprocessing complete.")