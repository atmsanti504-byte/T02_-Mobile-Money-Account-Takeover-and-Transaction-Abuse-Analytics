import pandas as pd
import numpy as np
from pathlib import Path

SEED = 2115799
rng = np.random.default_rng(SEED)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SCENARIO_DIR = DATA_DIR / "scenario"
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- Load Processed Data -------------------
accounts = pd.read_csv(PROCESSED_DIR / "accounts_clean.csv")
authentication = pd.read_csv(PROCESSED_DIR / "authentication_clean.csv")
transactions = pd.read_csv(PROCESSED_DIR / "transactions_clean.csv")
device_sim = pd.read_csv(PROCESSED_DIR / "device_sim_clean.csv")
geolocation = pd.read_csv(PROCESSED_DIR / "geolocation_clean.csv")
support = pd.read_csv(PROCESSED_DIR / "support_tickets_clean.csv")

# Convert timestamps
authentication["timestamp"] = pd.to_datetime(authentication["timestamp"])
transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
device_sim["timestamp"] = pd.to_datetime(device_sim["timestamp"])
geolocation["timestamp"] = pd.to_datetime(geolocation["timestamp"])
support["timestamp"] = pd.to_datetime(support["timestamp"])

print("=" * 60)
print("ATO SCENARIO INJECTION")
print("=" * 60)

# ------------------- Scenario Injection -------------------

# ----- Compromised accounts for the scenario
N_COMPROMISED_ACCOUNTS = 1000
FRAUD_TRANSACTIONS_PER_ACCOUNT = 5
compromised_accounts = rng.choice(accounts["account_id"].values,size=N_COMPROMISED_ACCOUNTS,replace=False)
print(f"\nCompromised accounts: {len(compromised_accounts)}")

# ----- Inject fraud transactions for compromised accounts
fraud_transaction_indices = []
for account_id in compromised_accounts:
    account_transactions = transactions[transactions["account_id"] == account_id]
    selected = rng.choice(account_transactions.index.values,size=FRAUD_TRANSACTIONS_PER_ACCOUNT,replace=False)
    fraud_transaction_indices.extend(selected)
    transactions["is_fraud"] = 0
    transactions["scenario_type"] = "LEGITIMATE"
    transactions.loc[fraud_transaction_indices,"is_fraud"] = 1

    # assign ato pattern
    scenario_types = ["SIM_SWAP_ATO","DEVICE_TAKEOVER","CREDENTIAL_ATTACK","TRAVEL_CONTEXT_ATO"]
    scenario_weights = [0.35, 0.30, 0.25, 0.10]
    transactions.loc[fraud_transaction_indices,
                     "scenario_type"] = rng.choice(scenario_types,size=len(fraud_transaction_indices),p=scenario_weights)

print("\nTransaction labels:")
print(transactions["is_fraud"].value_counts())

print("\nScenario types:")
print(
    transactions.loc[
        transactions["is_fraud"] == 1,
        "scenario_type"
    ].value_counts()
)

# --- modify fraud transaction behavior
fraud_mask = transactions["is_fraud"] == 1
# Higher transaction amounts
transactions.loc[fraud_mask, "amount"] = np.round(
    transactions.loc[fraud_mask, "amount"]* rng.uniform(1.8, 4.5, size=fraud_mask.sum()),2)
# Fraudulent activity favours higher-risk transaction types
transactions.loc[fraud_mask, "transaction_type"] = rng.choice(
    ["P2P_TRANSFER", "CASH_OUT", "BANK_TRANSFER"],size=fraud_mask.sum(),p=[0.45, 0.40, 0.15])
# Fraudulent activity uses channels differently
transactions.loc[fraud_mask, "channel"] = rng.choice(
    ["USSD", "Web", "Mobile_App"],size=fraud_mask.sum(),p=[0.20, 0.25, 0.55])

# ----- Suspicious authentication activity for compromised accounts
auth_injections = []

for account_id in compromised_accounts:
    fraud_events = transactions[
        (transactions["account_id"] == account_id) &
        (transactions["is_fraud"] == 1)
    ].sort_values("timestamp")

    first_fraud = fraud_events.iloc[0]
    base_time = first_fraud["timestamp"]

    for attempt in range(3):
        auth_injections.append({
            "event_id": f"ATO_AUTH_{account_id}_{attempt + 1}",
            "timestamp": base_time - pd.Timedelta(
                minutes=int(rng.integers(2, 15))
            ),
            "account_id": account_id,
            "ip_city": rng.choice([
                "Windhoek",
                "Swakopmund",
                "Walvis Bay",
                "Oshakati",
                "Rundu",
                "Keetmanshoop"
            ]),
            "device_id": f"ATO_DEV_{account_id}",
            "login_result": "FAILED",
            "mfa_used": False,
            "failure_reason": rng.choice([
                "Bad OTP",
                "Wrong Password",
                "Account Locked"
            ]),
            "login_hour": base_time.hour,
            "login_day": base_time.day_name(),
            "is_weekend": base_time.dayofweek >= 5,
            "scenario_label": "ATO"
        })
auth_injections = pd.DataFrame(auth_injections)
authentication["scenario_label"] = "LEGITIMATE"
authentication = pd.concat([authentication, auth_injections],ignore_index=True)
authentication = authentication.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
print(f"\nInjected authentication events: {len(auth_injections)}")

# ---- Inject Device/SIM changes for compromised accounts
device_injections = []
for account_id in compromised_accounts:
    fraud_events = transactions[
        (transactions["account_id"] == account_id) &
        (transactions["is_fraud"] == 1)
    ].sort_values("timestamp")
    
    first_fraud = fraud_events.iloc[0]
    base_time = first_fraud["timestamp"]
    scenario = first_fraud["scenario_type"]

    if scenario == "SIM_SWAP_ATO":
        change_type = "SIM_CHANGE"
    else:
        change_type = "DEVICE_CHANGE"
    device_injections.append({
        "event_id": f"ATO_DEV_EVT_{account_id}",
        "timestamp": base_time - pd.Timedelta(
            minutes=int(rng.integers(1, 10))
        ),
        "account_id": account_id,
        "change_type": change_type,
        "device_id": f"ATO_DEV_{account_id}",
        "sim_hash": f"ATO_SIM_{account_id}",
        "device_model": rng.choice([
            "Samsung_A15",
            "Samsung_A25",
            "iPhone_13",
            "iPhone_14",
            "Huawei_Nova",
            "Xiaomi_Redmi",
            "Google_Pixel"
        ]),
        "os_version": rng.choice([
            "Android_13",
            "Android_14",
            "Android_15",
            "iOS_17",
            "iOS_18"
        ]),
        "scenario_label": "ATO"
    })
device_injections = pd.DataFrame(device_injections)
device_sim["scenario_label"] = "LEGITIMATE"
device_sim = pd.concat([device_sim, device_injections],ignore_index=True)
device_sim = device_sim.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
print(f"Injected device/SIM events: {len(device_injections)}")


# ---- Inject unusual geolocation activity for compromised accounts
geo_injections = []
for account_id in compromised_accounts:
    fraud_events = transactions[
        (transactions["account_id"] == account_id) &
        (transactions["is_fraud"] == 1)].sort_values("timestamp")
    
    first_fraud = fraud_events.iloc[0]
    base_time = first_fraud["timestamp"]
    home_city = accounts.loc[accounts["account_id"] == account_id,"home_city"].iloc[0]

    alternative_cities = [
        city for city in [
            "Windhoek",
            "Swakopmund",
            "Walvis Bay",
            "Oshakati",
            "Rundu",
            "Keetmanshoop"
        ]
        if city != home_city
    ]
    geo_injections.append({
        "event_id": f"ATO_GEO_{account_id}",
        "timestamp": base_time - pd.Timedelta(
            minutes=int(rng.integers(1, 5))
        ),
        "account_id": account_id,
        "city": rng.choice(alternative_cities),
        "device_id": f"ATO_DEV_{account_id}",
        "scenario_label": "ATO"
    })

geo_injections = pd.DataFrame(geo_injections)
geolocation["scenario_label"] = "LEGITIMATE"
geolocation = pd.concat([geolocation, geo_injections],ignore_index=True)
geolocation = geolocation.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
print(f"Injected geolocation events: {len(geo_injections)}")

# ----- Support tickets related to ATO/CUSTOMER COMPLAINTS
support_injections = []

for account_id in compromised_accounts:
    fraud_events = transactions[
        (transactions["account_id"] == account_id) &
        (transactions["is_fraud"] == 1)
    ].sort_values("timestamp")

    fraud_event = fraud_events.iloc[-1]

    support_injections.append({
        "ticket_id": f"ATO_TICKET_{account_id}",
        "timestamp": fraud_event["timestamp"] + pd.Timedelta(
            minutes=int(rng.integers(5, 60))
        ),
        "account_id": account_id,
        "text": rng.choice([
            "There was a transaction on my account that I did not make.",
            "I noticed a transfer that I do not recognise.",
            "I received an OTP that I did not request.",
            "My SIM card was replaced and I cannot log in.",
            "My phone was lost and I need to secure my account."
        ]),
        "category": "Unauthorised Transaction",
        "priority": rng.choice(["High", "Medium"], p=[0.8, 0.2]),
        "scenario_label": "ATO"
    })

support_injections = pd.DataFrame(support_injections)
support["scenario_label"] = "LEGITIMATE"
support = pd.concat([support, support_injections],ignore_index=True)
support = support.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
print(f"Injected support tickets: {len(support_injections)}")

## Save the modified datasets with scenario injections
transactions.to_csv(SCENARIO_DIR / "transactions_scenario.csv", index=False)
authentication.to_csv(SCENARIO_DIR / "authentication_scenario.csv", index=False)
support.to_csv(SCENARIO_DIR / "support_scenario.csv", index=False)
device_sim.to_csv(SCENARIO_DIR / "device_sim_scenario.csv", index=False)
geolocation.to_csv(SCENARIO_DIR / "geolocation_scenario.csv", index=False)





    