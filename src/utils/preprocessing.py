import pandas as pd
import re
import string
import os
import matplotlib.pyplot as plt

SWEAR_WORDS = {
    "arse", "ass", "asshole", "bastard", "bitch", "bollocks", "bugger",
    "bullshit", "crap", "cunt", "damn", "dick", "douche", "dyke",
    "fag", "faggot", "fuck", "fucking", "goddamn", "hell", "horseshit",
    "jackass", "jerk", "motherfucker", "nigga", "nigger", "piss", "prick",
    "pussy", "shit", "shitty", "slut", "twat", "wanker"
}

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    
    text = text.lower()
    
    CONTRACTIONS = {
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that'd": "that would",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "would've": "would have",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)
    
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # remove repeated chars
    text = re.sub(r"http\S+", "", text)       # remove urls
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def extract_features(df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
    df["char_count"] = df[col].apply(len)
    df["word_count"] = df[col].apply(lambda x: len(x.split()))
    df["num_uppercase"] = df[col].apply(lambda x: sum(1 for c in x if c.isupper()))
    df["num_exclamation"] = df[col].apply(lambda x: x.count("!"))
    df["num_question"] = df[col].apply(lambda x: x.count("?"))
    df["capital_ratio"] = df[col].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x),1))
    df["has_swear"] = df[col].apply(lambda x: int(any(word in x.split() for word in SWEAR_WORDS)))
    return df


def prepare_jigsaw_multilevel(path: str):
    df = pd.read_csv(path)
    
    df['text'] = df['comment_text'].apply(clean_text)
    df = df[df['text'].str.strip() != ""].reset_index(drop=True)
    
    df['score'] = (
    df['toxic'] * 2 +
    df['severe_toxic'] * 4 +
    df['obscene'] * 2 +
    df['threat'] * 4 +
    df['insult'] * 2 +
    df['identity_hate'] * 3
)

    def map_label(score):
        if score <= 1:
            return "safe"
        elif 2 <= score <= 4:
            return "mild"
        elif 5 <= score <= 8:
            return "toxic"
        else:
            return "severe"

    df['label'] = df['score'].apply(map_label)

    label_counts = df['label'].value_counts().reindex(['safe','mild','toxic','severe'], fill_value=0)
    print("Number of comments per category:")
    print(label_counts)

    df = extract_features(df, col="text")
    
    cols_to_keep = ["text", "label", "score", "char_count", "word_count",
                    "num_uppercase", "num_exclamation", "num_question",
                    "capital_ratio", "has_swear"]
    df = df[cols_to_keep]
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/jigsaw_multilevel_features.csv", index=False)
    return df

if __name__ == "__main__":
    path = "data/raw/jigsaw_toxic_comments/train.csv"
    df = prepare_jigsaw_multilevel(path)
    print("Processed dataset saved to data/processed/jigsaw_multilevel_features.csv")
    print(df.head())
