from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURE_PATH = DATA_DIR / "features" / "ato_features.csv"
df = pd.read_csv(FEATURE_PATH)
print("=" * 60)
print("UNSUPERVISED ANOMALY DETECTION - ISOLATION FOREST")
print("=" * 60)
print(f"Dataset shape: {df.shape}")

# clean features
transaction_ids = df["transaction_id"].copy()
# Remove identifiers / duplicate feature
drop_columns = ["transaction_id","is_fraud","typical_daily_transactions_x"]
df = df.drop(columns=[c for c in drop_columns if c in df.columns])
if "typical_daily_transactions_y" in df.columns:
    df = df.rename(columns={"typical_daily_transactions_y":
                            "typical_daily_transactions"})
X = df.copy()
y_true = pd.read_csv(FEATURE_PATH)["is_fraud"]
print(f"Anomaly features: {X.shape[1]}")
print(f"Records: {len(X)}")

# isolation forest
# The synthetic dataset contains 5% fraud.
# contamination therefore provides an initial anomaly threshold.
CONTAMINATION = 0.05
model = IsolationForest(
    n_estimators=300,
    contamination=CONTAMINATION,
    random_state=2115799,
    n_jobs=-1
)

print("\nTraining Isolation Forest...")
model.fit(X)
print("Training complete.")

# Isolation Forest:-1 = anomaly 1 = normal
predictions = model.predict(X)

# Convert to: 1 = anomaly 0 = normal
anomaly = (predictions == -1).astype(int)
# Higher anomaly_score = more anomalous
anomaly_score = -model.decision_function(X)
print("\n" + "=" * 60)
print("ANOMALY DETECTION RESULTS")
print("=" * 60)
print(f"Anomalies detected: {anomaly.sum()}")
print(f"Anomaly percentage: {anomaly.mean() * 100:.2f}%")
precision = precision_score(y_true,anomaly,zero_division=0)
recall = recall_score(y_true,anomaly,zero_division=0)
f1 = f1_score(y_true,anomaly,zero_division=0)
print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# analyst threshold
threshold = np.percentile(anomaly_score,100 * (1 - CONTAMINATION))
print("\n" + "=" * 60)
print("ANOMALY THRESHOLD")
print("=" * 60)
print(f"Threshold: {threshold:.6f}")
print("Transactions with anomaly_score >= threshold ""are flagged for analyst review.")

results = pd.DataFrame({
    "transaction_id": transaction_ids,
    "is_fraud": y_true,
    "anomaly_score": anomaly_score,
    "is_anomaly": anomaly
})
results = results.sort_values("anomaly_score",ascending=False)
results.to_csv(DATA_DIR / "isolation_forest_results.csv",index=False)

# top anomalies
print("\nTop 15 anomalous transactions:")
print(results.head(15).to_string(index=False))

#anomaly score distribution
plt.figure(figsize=(10, 6))
plt.hist(anomaly_score,bins=50)
plt.axvline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)
plt.title("Isolation Forest Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Number of Transactions")
plt.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / "isolation_forest_anomaly_distribution.png",dpi=300)
plt.close()

print("\n" + "=" * 60)
print("ISOLATION FOREST COMPLETE")
print("=" * 60)
print("Saved:")
print("- data/isolation_forest_results.csv")
print("- data/isolation_forest_anomaly_distribution.png")











