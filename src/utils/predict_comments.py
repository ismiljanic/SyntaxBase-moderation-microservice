import joblib
from scipy.sparse import hstack
import numpy as np
import re

SWEAR_WORDS = {
    "arse", "ass", "asshole", "bastard", "bitch", "bollocks", "bugger",
    "bullshit", "crap", "cunt", "damn", "dick", "douche", "dyke",
    "fag", "faggot", "fuck", "fucking", "goddamn", "hell", "horseshit",
    "jackass", "jerk", "motherfucker", "nigga", "nigger", "piss", "prick",
    "pussy", "shit", "shitty", "slut", "twat", "wanker"
}

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_numeric_features_single(text):
    char_count = len(text)
    word_count = len(text.split())
    num_uppercase = sum(1 for c in text if c.isupper())
    num_exclamation = text.count("!")
    num_question = text.count("?")
    has_swear = int(any(word in text.split() for word in SWEAR_WORDS))
    return [char_count, word_count, num_uppercase, num_exclamation, num_question, has_swear]


model = joblib.load("models/saved/xgboost.pkl")
vectorizer = joblib.load("models/saved/vectorizer.pkl")
le = joblib.load("models/saved/label_encoder.pkl")

print("Comment Moderation Interactive Tool")
print("Type your comment and press Enter. Type 'exit' to quit.\n")

while True:
    comment = input("Your comment: ")
    if comment.lower() == "exit":
        break

    comment_clean = clean_text(comment)
    numeric_features = np.array([add_numeric_features_single(comment_clean)])
    text_vector = vectorizer.transform([comment_clean])
    combined_features = hstack([text_vector, numeric_features])

    pred_enc = model.predict(combined_features)
    pred_label = le.inverse_transform(pred_enc)[0]

    print(f"Predicted label: {pred_label}\n")
