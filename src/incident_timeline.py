from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCENARIO_DIR = DATA_DIR / "scenario"

transactions = pd.read_csv(SCENARIO_DIR / "transactions_scenario.csv")
authentication = pd.read_csv(SCENARIO_DIR / "authentication_scenario.csv")
device = pd.read_csv( SCENARIO_DIR / "device_sim_scenario.csv")
geolocation = pd.read_csv(SCENARIO_DIR / "geolocation_scenario.csv")
support = pd.read_csv(SCENARIO_DIR / "support_scenario.csv")

print("=" * 60)
print("INCIDENT TIMELINE + EVIDENCE CORRELATION")
print("=" * 60)

# find fraud case
fraud_transactions = transactions[transactions["is_fraud"] == 1].copy()
# Select first fraudulent transaction
fraud_case = fraud_transactions.iloc[0]
account_id = fraud_case["account_id"]
fraud_transaction_id = fraud_case["transaction_id"]
print(f"\nSelected account: {account_id}")
print(f"Fraud transaction: {fraud_transaction_id}")

def find_timestamp_column(df):
    candidates = [
        "timestamp",
        "event_timestamp",
        "transaction_timestamp",
        "login_timestamp",
        "event_time",
        "created_at",
        "ticket_timestamp"
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(f"No timestamp column found. Available columns: {list(df.columns)}")

# accoun event collection
events = []
def add_events(df, source, event_id_candidates):
    """Add account events to common timeline structure."""
    account_events = df[df["account_id"] == account_id].copy()

    if account_events.empty:
        return
    timestamp_column = find_timestamp_column(account_events)
    # Convert timestamp
    account_events[timestamp_column] = pd.to_datetime(
        account_events[timestamp_column],
        errors="coerce"
    )
    # Find event ID
    event_id_column = None
    for candidate in event_id_candidates:
        if candidate in account_events.columns:
            event_id_column = candidate
            break

    if event_id_column is None:
        event_ids = range(len(account_events))
    else:
        event_ids = account_events[event_id_column]

    for idx, row in account_events.iterrows():
        event_description = source
        # Add useful contextual information
        if source == "AUTHENTICATION":
            event_description = (
                f"Login {row.get('login_result', '')}")
            if "failure_reason" in row:
                reason = row["failure_reason"]

                if pd.notna(reason) and reason != "Not_Applicable":
                    event_description += f" - {reason}"
        elif source == "DEVICE_SIM":
            event_description = (f"Device/SIM event: "f"{row.get('change_type', '')}")
        elif source == "GEOLOCATION":event_description = (f"Location: {row.get('city', '')}")
        elif source == "TRANSACTION":
            event_description = (
                f"Transaction: "
                f"{row.get('transaction_type', '')} "
                f"amount={row.get('amount', '')}")

        elif source == "SUPPORT":
            text = str(row.get("ticket_text",row.get("text", "")))
            event_description = (f"Support ticket: {text[:120]}")
        events.append({
            "account_id": account_id,
            "timestamp": row[timestamp_column],
            "source": source,
            "event_id": (
                row[event_id_column]
                if event_id_column
                else str(event_ids[idx])
            ),"event_description": event_description})

add_events(authentication,"AUTHENTICATION",["event_id", "authentication_id"])
add_events(device,"DEVICE_SIM",["event_id", "device_event_id"])
add_events(geolocation,"GEOLOCATION",["event_id", "geolocation_id"])
add_events(transactions,"TRANSACTION",["transaction_id", "event_id"])
add_events(support,"SUPPORT",["ticket_id", "event_id"])

timeline = pd.DataFrame(events)
timeline = timeline.dropna(subset=["timestamp"])
timeline = timeline.sort_values("timestamp").reset_index(drop=True)

fraud_timestamp_column = find_timestamp_column(transactions)
fraud_time = pd.to_datetime(fraud_case[fraud_timestamp_column])
# Focus investigation on 24 hours before and 6 hours after the suspicious transaction.
start_time = fraud_time - pd.Timedelta(hours=24)
end_time = fraud_time + pd.Timedelta(hours=6)
timeline = timeline[(timeline["timestamp"] >= start_time) &(timeline["timestamp"] <= end_time)].copy()
timeline.insert(0,"sequence",range(1, len(timeline) + 1))
output_path = DATA_DIR / "incident_timeline.csv"
timeline.to_csv(output_path,index=False)

## validation
print("\n" + "=" * 60)
print("INCIDENT TIMELINE")
print("=" * 60)

print(f"Investigation window:")
print(f"  Start: {start_time}")
print(f"  End:   {end_time}")

print(f"\nEvidence events: {len(timeline)}")

print("\nEvents by source:")
print(
    timeline["source"]
    .value_counts()
    .to_string()
)

print("\nChronological evidence:")
print(
    timeline[
        [
            "sequence",
            "timestamp",
            "source",
            "event_id",
            "event_description"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("INCIDENT TIMELINE COMPLETE")
print("=" * 60)

print(f"Saved: {output_path}")






































