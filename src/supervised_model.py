import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

## ----- Load feature
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURE_PATH = DATA_DIR / "features" / "ato_features.csv"
df = pd.read_csv(FEATURE_PATH)
print("=" * 60)
print("SUPERVISED MODEL - XGBOOST")
print("=" * 60)
print(f"Dataset shape: {df.shape}")

# //clean feature table
# Remove identifier
if "transaction_id" in df.columns:df = df.drop(columns=["transaction_id"])

# Remove duplicate feature caused by merging
if "typical_daily_transactions_x" in df.columns:df = df.drop(columns=["typical_daily_transactions_x"])

if "typical_daily_transactions_y" in df.columns:
    df = df.rename(columns={"typical_daily_transactions_y": "typical_daily_transactions"})

TARGET = "is_fraud"
X = df.drop(columns=[TARGET])
y = df[TARGET]
print(f"Features: {X.shape[1]}")
print(f"Target distribution:")
print(y.value_counts())
print()


## Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=2115799,
    stratify=y
)
print("Train/Test split:")
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")
print()

## XGBOOST MODEL
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive
print(f"Scale positive weight: {scale_pos_weight:.2f}")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=2115799,
    n_jobs=-1
)

# train
print("\nTraining XGBoost...")
model.fit(X_train,y_train)
print("Training complete.")

# / predicitons
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# eval
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(cm)

# save conf matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Legitimate", "Fraud"]).plot(ax=ax)
plt.title("XGBoost ATO Fraud Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig(DATA_DIR / "xgboost_confusion_matrix.png",dpi=300)
plt.close()

## feature NB
importance = pd.DataFrame({"feature": X.columns,"importance": model.feature_importances_})
importance = importance.sort_values("importance",ascending=False)
print("\nTop 15 Features:")
print(importance.head(15).to_string(index=False))
importance.to_csv(DATA_DIR / "xgboost_feature_importance.csv",index=False)

model.save_model(DATA_DIR / "xgboost_ato_model.json")
print("\n" + "=" * 60)
print("XGBOOST MODEL COMPLETE")
print("=" * 60)
print("Saved:")
print("- data/xgboost_confusion_matrix.png")
print("- data/xgboost_feature_importance.csv")
print("- data/xgboost_ato_model.json")