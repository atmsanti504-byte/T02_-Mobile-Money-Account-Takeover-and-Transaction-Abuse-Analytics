"""
Text-mining / NLP workflow on synthetic customer-support tickets.

- TF-IDF + Logistic Regression classifier -> ticket category
  (preliminary proxy for "is this complaint fraud-related?")
- Simple rule-based indicator/entity extraction (keyword and pattern based)
  as a first-pass intelligence-enrichment step.
"""
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "outputs"

FRAUD_KEYWORDS = [
    "did not make", "without permission", "sim card", "otp", "pin",
    "missing", "unauthorised", "unauthorized", "changed my", "access",
]


def extract_indicators(text: str):
    text_l = text.lower()
    hits = [kw for kw in FRAUD_KEYWORDS if kw in text_l]
    has_amount_ref = bool(re.search(r"\b(balance|amount|money|charged)\b", text_l))
    return {"keyword_hits": hits, "n_hits": len(hits), "amount_reference": has_amount_ref}


def main():
    tickets = pd.read_csv(DATA / "support_tickets.csv")

    # indicator extraction pass
    indicators = tickets["text"].apply(extract_indicators).apply(pd.Series)
    tickets = pd.concat([tickets, indicators], axis=1)
    tickets["likely_fraud_related"] = (tickets["n_hits"] >= 1).astype(int)

    # classifier: text -> category
    X_train, X_test, y_train, y_test = train_test_split(
        tickets["text"], tickets["category"], test_size=0.25, random_state=42,
        stratify=tickets["category"],
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    macro_f1 = round(f1_score(y_test, preds, average="macro"), 3)

    result = {
        "macro_f1": macro_f1,
        "per_class_f1": {k: round(v["f1-score"], 3) for k, v in report.items()
                          if k in tickets["category"].unique()},
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_tickets_flagged_by_keywords": int(tickets["likely_fraud_related"].sum()),
        "sample_indicator_extractions": tickets.loc[
            tickets["likely_fraud_related"] == 1, ["ticket_id", "category", "keyword_hits"]
        ].head(5).to_dict(orient="records"),
    }

    tickets.to_csv(OUT / "tickets_enriched.csv", index=False)
    with open(OUT / "text_mining_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("[text-mining] macro F1:", macro_f1)
    print("[text-mining] per-class F1:", result["per_class_f1"])
    print("[text-mining] keyword-flagged tickets:", result["n_tickets_flagged_by_keywords"])


if __name__ == "__main__":
    main()
