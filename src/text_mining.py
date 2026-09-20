from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "support_tickets.csv"
SCENARIO_PATH = DATA_DIR / "scenario" / "support_scenario.csv"

# Prefer scenario data because it contains the ATO-labelled cases.
if SCENARIO_PATH.exists():
    tickets = pd.read_csv(SCENARIO_PATH)
else:
    tickets = pd.read_csv(RAW_PATH)
print("=" * 60)
print("TEXT MINING / NLP - SUPPORT TICKETS")
print("=" * 60)
print(f"Dataset shape: {tickets.shape}")
print("\nColumns:")
print(tickets.columns.tolist())

# ID text column
text_candidates = [
    "ticket_text",
    "text",
    "description",
    "message",
    "content"
]
TEXT_COLUMN = None
for column in text_candidates:
    if column in tickets.columns:
        TEXT_COLUMN = column
        break
if TEXT_COLUMN is None:
    raise ValueError(
        f"No text column found. Available columns: "
        f"{tickets.columns.tolist()}"
    )
print(f"\nText column: {TEXT_COLUMN}")

# ID label
label_candidates = [
    "scenario_label",
    "label",
    "is_fraud",
    "category"
]
LABEL_COLUMN = None
for column in label_candidates:
    if column in tickets.columns:
        LABEL_COLUMN = column
        break
if LABEL_COLUMN is None:
    raise ValueError(
        f"No suitable label column found. Available columns: "
        f"{tickets.columns.tolist()}"
    )
print(f"Label column: {LABEL_COLUMN}")

# preprocesasing
tickets[TEXT_COLUMN] = (
    tickets[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
tickets = tickets[tickets[TEXT_COLUMN].str.len() > 0].copy()
print(f"\nUsable text records: {len(tickets)}")

# label normalisation
print("\nOriginal labels:")
print(tickets[LABEL_COLUMN].value_counts())

def normalise_label(value):

    value = str(value).upper().strip()

    if value in ["ATO", "FRAUD", "1", "TRUE"]:
        return 1

    if value in ["LEGITIMATE", "NORMAL", "0", "FALSE"]:
        return 0

    return None

tickets["target"] = tickets[LABEL_COLUMN].apply(normalise_label)
# Remove records that cannot be mapped
tickets = tickets.dropna(subset=["target"])
tickets["target"] = tickets["target"].astype(int)
print("\nNormalised labels:")
print(tickets["target"].value_counts())

# train_test_split
X_text = tickets[TEXT_COLUMN]
y = tickets["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=2115799,
    stratify=y
)
print("\nTrain/Test:")
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")

# tf-idf
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("\nTF-IDF:")
print(f"Training matrix: {X_train_tfidf.shape}")
print(f"Testing matrix:  {X_test_tfidf.shape}")

# classifier
classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=2115799
)
print("\nTraining text classifier...")
classifier.fit(X_train_tfidf,y_train)
print("Training complete.")

#predictions
y_pred = classifier.predict(X_test_tfidf)

# eval
precision = precision_score(y_test,y_pred,zero_division=0)
recall = recall_score(y_test,y_pred,zero_division=0)
f1 = f1_score(y_test,y_pred,zero_division=0)
cm = confusion_matrix(y_test,y_pred)

print("\n" + "=" * 60)
print("TEXT CLASSIFICATION RESULTS")
print("=" * 60)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test,y_pred,zero_division=0 ))
print("Confusion Matrix:")
print(cm)

# NB ATO INDICATORS 
terms = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]
term_importance = pd.DataFrame({"term": terms,"coefficient": coefficients})
term_importance["absolute_coefficient"] = (term_importance["coefficient"].abs())
term_importance = term_importance.sort_values("absolute_coefficient", ascending=False)
print("\nTop text indicators:")
print(term_importance.head(20)[["term", "coefficient"]].to_string(index=False))

term_importance.to_csv(DATA_DIR / "text_indicator_importance.csv",index=False)
pd.DataFrame(cm,
    index=["Actual Legitimate", "Actual ATO"],
    columns=["Predicted Legitimate", "Predicted ATO"]
).to_csv(DATA_DIR / "text_confusion_matrix.csv")

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Support Ticket Text Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1],["Legitimate", "ATO"])

plt.yticks([0, 1],["Legitimate", "ATO"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(DATA_DIR / "text_classification_confusion_matrix.png",dpi=300)
plt.close()

print("\n" + "=" * 60)
print("TEXT MINING COMPLETE")
print("=" * 60)
print("Saved:")
print("- data/text_indicator_importance.csv")
print("- data/text_confusion_matrix.csv")
print("- data/text_classification_confusion_matrix.png")

