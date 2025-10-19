import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# -----------------------------
# Text Cleaning & Feature Engineering
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_numeric_features(df, col="text"):
    df["char_count"] = df[col].apply(len)
    df["word_count"] = df[col].apply(lambda x: len(x.split()))
    df["num_uppercase"] = df[col].apply(lambda x: sum(1 for c in x if c.isupper()))
    df["num_exclamation"] = df[col].apply(lambda x: x.count("!"))
    df["num_question"] = df[col].apply(lambda x: x.count("?"))
    return df

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/jigsaw_multilevel_features.csv")
df = df.dropna(subset=["text"])
df["text"] = df["text"].apply(clean_text)
df = add_numeric_features(df)

# SAMPLE_SIZE = 40000
# if len(df) > SAMPLE_SIZE:
    # df = df.sample(n=SAMPLE_SIZE, random_state=42)


X = df["text"]
y = df["label"]

# Encode labels for XGBoost
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train_enc, y_temp_enc = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)
X_val, X_test, y_val_enc, y_test_enc = train_test_split(
    X_temp, y_temp_enc, test_size=0.5, stratify=y_temp_enc, random_state=42
)

# Compute weights for each sample
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train_enc
)

# -----------------------------
# TF-IDF Vectorization + Numeric Features
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,3),
    analyzer='word',
    sublinear_tf=True,
    min_df=3,
    max_df=0.9
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

numeric_cols = ["char_count", "word_count", "num_uppercase", "num_exclamation", "num_question", "has_swear"]
X_train_num = df.loc[X_train.index, numeric_cols].values
X_test_num  = df.loc[X_test.index, numeric_cols].values

X_train_combined = hstack([X_train_vec, X_train_num])
X_test_combined = hstack([X_test_vec, X_test_num])

# -----------------------------
# Models
# -----------------------------
models = {
    #  "Logistic Regression": LogisticRegression(
    #     max_iter=10000,
    #     class_weight="balanced",
    #     solver="saga",
    #     n_jobs=-1
    # ),
    # "Random Forest": RandomForestClassifier(
    #     n_estimators=100,
    #     max_depth=15,
    #     class_weight="balanced",
    #     random_state=42,
    #     n_jobs=-1
    # ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42
    )
}

# -----------------------------
# Train & Evaluate
# -----------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")

    if name == "XGBoost":
        model.fit(X_train_combined, y_train_enc, sample_weight=sample_weights)
        y_pred_enc = model.predict(X_test_combined)
        y_pred = le.inverse_transform(y_pred_enc)
        y_test_labels = le.inverse_transform(y_test_enc)
    else:
        model.fit(X_train_combined, le.inverse_transform(y_train_enc))
        y_pred = model.predict(X_test_combined)
        y_test_labels = le.inverse_transform(y_test_enc)

    print(f"\n===== {name} Classification Report =====")
    print(classification_report(y_test_labels, y_pred, digits=4))

    cm = confusion_matrix(y_test_labels, y_pred, labels=["safe", "mild", "toxic", "severe"])
    print(f"\n===== {name} Confusion Matrix =====\n{cm}")

# -----------------------------
# Save Models & Vectorizer
# -----------------------------
os.makedirs("models/saved/classical", exist_ok=True)
for name, model in models.items():
    file_name = f"models/saved/classical/{name.lower().replace(' ','_')}.pkl"
    joblib.dump(model, file_name)

joblib.dump(vectorizer, "models/saved/classical/vectorizer.pkl")
joblib.dump(le, "models/saved/classical/label_encoder.pkl")
print("\nModels, vectorizer, and label encoder saved in models/saved/")