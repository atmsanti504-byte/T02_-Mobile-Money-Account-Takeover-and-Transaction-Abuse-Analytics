"""
Synthetic data generator for the Mobile Money Account Takeover capstone.

Produces a PROTOTYPE-SCALE sample (thousands, not millions, of records) of:
  - authentication events
  - transactions
  - device/SIM-change events
  - customer-support tickets (free text)

All data is entirely synthetic. No real customer, transaction or device
records are used or referenced. A fixed random seed is used so results
are reproducible.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 2115799
rng = np.random.default_rng(SEED)

N_ACCOUNTS = 800
N_AUTH_EVENTS = 12000
N_TXNS = 9000
N_DEVICE_EVENTS = 1500
N_TICKETS = 1200

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(exist_ok=True)

CITIES = [
    ("Windhoek", -22.5597, 17.0832), 
    ("Swakopmund", -22.6784, 14.5268), 
    ("Walvis Bay", -22.9575, 14.5053), 
    ("Oshakati", -17.7833, 15.6833), 
    ("Rundu", -17.9333, 19.7667), 
    ("Keetmanshoop", -26.5786, 18.1333),
]

account_ids = np.array([f"ACC{100000+i}" for i in range(N_ACCOUNTS)])
# ~6% of accounts are "compromised" at some point in the observation window
compromised_accounts = set(rng.choice(account_ids, size=int(N_ACCOUNTS * 0.06), replace=False))


def home_city_for(acc):
    idx = hash(acc) % len(CITIES)
    return CITIES[idx]


# ---------------------------------------------------------------- AUTH LOG
def generate_auth_events():
    rows = []
    for i in range(N_AUTH_EVENTS):
        acc = rng.choice(account_ids)
        is_takeover = acc in compromised_accounts and rng.random() < 0.35
        city, lat, lon = home_city_for(acc)
        if is_takeover:
            lat += rng.normal(0, 3.5)
            lon += rng.normal(0, 3.5)
            login_result = rng.choice(["success", "failed"], p=[0.55, 0.45])
            mfa_flag = rng.choice([0, 1], p=[0.7, 0.3])
            failure_reason = rng.choice(["bad_otp", "device_mismatch", "none"], p=[0.5, 0.3, 0.2])
        else:
            lat += rng.normal(0, 0.05)
            lon += rng.normal(0, 0.05)
            login_result = rng.choice(["success", "failed"], p=[0.93, 0.07])
            mfa_flag = rng.choice([0, 1], p=[0.2, 0.8])
            failure_reason = rng.choice(["bad_password", "none"], p=[0.3, 0.7])

        rows.append({
            "event_id": f"AUTH{i:06d}",
            "account_id": acc,
            "timestamp": pd.Timestamp("2026-06-01") + pd.to_timedelta(rng.integers(0, 60 * 24 * 60), unit="m"),
            "device_id": f"DEV{hash((acc, is_takeover)) % 5000:05d}",
            "login_result": login_result,
            "mfa_flag": mfa_flag,
            "failure_reason": failure_reason,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "is_account_takeover": int(is_takeover),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- TXNS
def generate_transactions():
    rows = []
    channels = ["p2p_transfer", "merchant_payment", "cash_out", "bill_payment"]
    for i in range(N_TXNS):
        acc = rng.choice(account_ids)
        is_fraud = acc in compromised_accounts and rng.random() < 0.4
        if is_fraud:
            amount = float(rng.lognormal(mean=8.5, sigma=1.0))  # larger, more variable
            channel = rng.choice(["cash_out", "p2p_transfer"], p=[0.6, 0.4])
        else:
            amount = float(rng.lognormal(mean=6.0, sigma=0.8))
            channel = rng.choice(channels, p=[0.4, 0.35, 0.15, 0.1])

        rows.append({
            "transaction_id": f"TXN{i:06d}",
            "account_id": acc,
            "timestamp": pd.Timestamp("2026-06-01") + pd.to_timedelta(rng.integers(0, 60 * 24 * 60), unit="m"),
            "amount": round(amount, 2),
            "currency": "KES",
            "channel": channel,
            "counterparty_id": f"CP{rng.integers(0, 4000):05d}",
            "is_fraud": int(is_fraud),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- DEVICE
def generate_device_events():
    rows = []
    change_types = ["sim_change", "device_change", "os_update"]
    for i in range(N_DEVICE_EVENTS):
        acc = rng.choice(account_ids)
        is_suspicious = acc in compromised_accounts and rng.random() < 0.5
        change_type = "sim_change" if is_suspicious and rng.random() < 0.7 else rng.choice(change_types)
        rows.append({
            "event_id": f"DEVCH{i:06d}",
            "account_id": acc,
            "timestamp": pd.Timestamp("2026-06-01") + pd.to_timedelta(rng.integers(0, 60 * 24 * 60), unit="m"),
            "change_type": change_type,
            "device_model_hash": f"MODEL{rng.integers(0, 200):04d}",
            "is_suspicious": int(is_suspicious),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- TICKETS
TEMPLATES = {
    "account_takeover": [
        "I did not make this transfer, someone accessed my account without permission.",
        "My SIM card stopped working and now money is missing from my wallet.",
        "I received an OTP I never requested and then my balance dropped.",
        "Someone changed my PIN and I can no longer log in to my account.",
    ],
    "billing_dispute": [
        "I was charged twice for the same merchant payment this week.",
        "The transaction fee on my transfer looks higher than usual, please review.",
        "I was billed for a bill payment that never went through.",
    ],
    "technical_issue": [
        "The app keeps crashing when I try to send money to a contact.",
        "I cannot complete a cash-out at the agent, the app freezes on confirmation.",
        "USSD menu is not loading for me since this morning.",
    ],
    "general_inquiry": [
        "How long does a merchant payment normally take to reflect?",
        "Can you explain the daily transfer limit on my account tier?",
        "What documents do I need to upgrade my account tier?",
    ],
}


def generate_tickets():
    rows = []
    categories = list(TEMPLATES.keys())
    for i in range(N_TICKETS):
        acc = rng.choice(account_ids)
        weights = [0.30, 0.20, 0.25, 0.25] if acc in compromised_accounts else [0.06, 0.24, 0.35, 0.35]
        cat = rng.choice(categories, p=weights)
        text = rng.choice(TEMPLATES[cat])
        rows.append({
            "ticket_id": f"TCK{i:06d}",
            "account_id_hashed": f"H{hash(acc) % 100000:05d}",
            "timestamp": pd.Timestamp("2026-06-01") + pd.to_timedelta(rng.integers(0, 60 * 24 * 60), unit="m"),
            "text": text,
            "category": cat,
        })
    return pd.DataFrame(rows)


def main():
    auth = generate_auth_events()
    txns = generate_transactions()
    dev = generate_device_events()
    tickets = generate_tickets()

    auth.to_csv(OUT_DIR / "auth_events.csv", index=False)
    txns.to_csv(OUT_DIR / "transactions.csv", index=False)
    dev.to_csv(OUT_DIR / "device_events.csv", index=False)
    tickets.to_csv(OUT_DIR / "support_tickets.csv", index=False)

    print("Generated:")
    print(f"  auth_events.csv       {len(auth):>6} rows")
    print(f"  transactions.csv      {len(txns):>6} rows")
    print(f"  device_events.csv     {len(dev):>6} rows")
    print(f"  support_tickets.csv   {len(tickets):>6} rows")
    print(f"  compromised accounts (ground truth, for evaluation only): {len(compromised_accounts)}")


if __name__ == "__main__":
    main()
