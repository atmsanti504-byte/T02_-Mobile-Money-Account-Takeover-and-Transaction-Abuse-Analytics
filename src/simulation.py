from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SEED = 2115799
ITERATIONS = 1000
rng = np.random.default_rng(SEED)

transactions = pd.read_csv(
    DATA_DIR / "features" / "ato_features.csv"
)
fraud_transactions = transactions[
    transactions["is_fraud"] == 1
].copy()
fraud_amounts = fraud_transactions["amount"].values
print("=" * 60)
print("ATO RESPONSE CONTROL SIMULATION")
print("=" * 60)
print(f"Fraud transactions available: {len(fraud_transactions)}")
print(f"Monte Carlo iterations/scenario: {ITERATIONS}")


# ============================================================
# CONTROL SCENARIOS
# ============================================================

# Assumptions:
#
# 1. BASELINE:
#    Manual investigation with slower response.
#
# 2. RISK-BASED:
#    XGBoost risk prioritisation reduces response time
#    and improves the proportion of cases acted upon.
#
# 3. ENHANCED:
#    Multi-signal workflow combining supervised,
#    anomaly and text intelligence with automated
#    prioritisation.
#
# These are simulation assumptions.

scenarios = {
    "Baseline Manual": {
        "detection_rate": 0.60,
        "response_mean": 30,
        "response_std": 8,
    },
    "Risk-Based XGBoost": {
        "detection_rate": 0.937,
        "response_mean": 15,
        "response_std": 4,
    },
    "Enhanced Multi-Signal": {
        "detection_rate": 0.95,
        "response_mean": 8,
        "response_std": 2,
    },
}


# Simulation

results = []
for scenario_name, settings in scenarios.items():
    for iteration in range(1, ITERATIONS + 1):
        # Sample a batch of fraud cases
        sample_size = 100
        sampled_amounts = rng.choice(
            fraud_amounts,
            size=sample_size,
            replace=True
        )

        # Determine which cases are detected
        detected = (rng.random(sample_size)< settings["detection_rate"])
        detected_amounts = sampled_amounts[detected]

        # Simulated analyst response time
        response_times = rng.normal(
            settings["response_mean"],
            settings["response_std"],
            size=len(detected_amounts))

        # Prevent negative response times
        response_times = np.maximum(response_times,1)

        # Define 15-minute operational SLA
        sla_breaches = response_times > 15
        results.append({
            "scenario": scenario_name,
            "iteration": iteration,
            "cases_simulated": sample_size,
            "cases_detected": len(detected_amounts),
            "detection_rate": (
                len(detected_amounts) / sample_size
            ),
            "mean_response_minutes": (
                response_times.mean()
                if len(response_times) > 0
                else 0
            ),
            "p95_response_minutes": (
                np.percentile(response_times, 95)
                if len(response_times) > 0
                else 0
            ),
            "sla_breaches": sla_breaches.sum(),
            "sla_breach_rate": (
                sla_breaches.mean()
                if len(response_times) > 0
                else 0
            ),
            "amount_under_review": (
                detected_amounts.sum()
            ),
        })


results_df = pd.DataFrame(results)
summary = (
    results_df
    .groupby("scenario")
    .agg(
        detection_rate=("detection_rate", "mean"),
        mean_response_minutes=(
            "mean_response_minutes",
            "mean"
        ),
        p95_response_minutes=(
            "p95_response_minutes",
            "mean"
        ),
        mean_sla_breach_rate=(
            "sla_breach_rate",
            "mean"
        ),
        mean_amount_under_review=(
            "amount_under_review",
            "mean"
        ),
    )
    .reset_index()
)


# Convert rates to percentages
summary["detection_rate"] *= 100
summary["mean_sla_breach_rate"] *= 100

print("\n" + "=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)
print(summary.to_string(index=False,float_format=lambda x: f"{x:.2f}"))

results_df.to_csv(DATA_DIR / "simulation_results.csv",index=False)
summary.to_csv(DATA_DIR / "simulation_summary.csv",index=False)

print("\n" + "=" * 60)
print("SIMULATION COMPLETE")
print("=" * 60)
print("Saved:")
print("- data/simulation_results.csv")
print("- data/simulation_summary.csv")