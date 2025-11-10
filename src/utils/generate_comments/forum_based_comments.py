#!/usr/bin/env python3
"""
generate_forum_comments.py

Generates a forum-specific test dataset CSV for toxicity/moderation models
with multiple acceptable labels per comment.
"""
import pandas as pd
import os

# ==============================
# Forum-specific examples
# ==============================
examples = [
    ("How do I center a div in CSS?", "safe", "safe", "Typical student question, completely safe"),
    ("I solved the exercise using a for loop!", "safe", "safe", "Sharing solution, safe"),
    ("Thanks for this tutorial, very helpful!", "safe", "safe", "Positive feedback"),
    ("I hate this error message!", "mild", "mild,safe", "Mild frustration, not really abusive"),
    ("You're doing it wrong", "mild", "mild, safe", "Minor criticism, no real toxicity"),
    ("This is so confusing lol", "mild", "mild, safe", "Joke + minor frustration"),
    ("Stop asking stupid questions", "toxic", "toxic, mild", "Rude, insulting another user"),
    ("Your code is garbage", "mild", "toxic, mild", "Harsh judgment of work"),
    ("This is nonsense", "mild", "mild, safe", "Dismissive comment"),
    ("I hope you fail your exams", "mild", "toxic, mild", "Exaggerated negative comment"),
    ("This is so bad, I want to smash my computer", "mild", "mild, safe", "Over-the-top exaggeration"),
    ("Can someone explain this function?", "safe", "safe", "Standard forum question"),
    ("I tried this solution and it worked!", "safe", "safe", "Sharing personal experience"),
    ("Thanks, that makes sense now", "safe", "safe", "Gratitude, positive engagement"),
    ("Why is this not working???", "safe", "safe, mild", "Mild frustration"),
    ("You're so slow, read the docs", "mild", "mild, safe, toxic", "Slightly aggressive insult"),
    ("This is pointless", "safe", "safe, mild", "Dismissive, could discourage others"),
    ("Help! I'm stuck on the exercise", "safe", "safe", "Asking for help, safe"),
    ("What are you thinking you little faggot i hope your whole family burns alive. Fuck you and your little forum SyntaxBase jews.", "severe", "severe,toxic", "Serious racial and gender slurs."),
    ("Lol, this tutorial is terrible", "mild", "mild", "Mild joke / frustration"),
    ("This is absolute trash", "mild", "toxic, mild", "Strong negative opinion"),
    ("I'm going to delete my account over this", "safe", "safe", "Exaggerated response, but safe"),
    ("I love coding here!", "safe", "safe", "Positive engagement"),
    ("The instructions are so confusing", "safe", "safe, mild", "Mild criticism"),
    ("You clearly didn't read the tutorial", "safe", "safe, mild", "Small indirect insult toward user"),
    ("Thanks everyone for the feedback", "safe", "safe", "Positive engagement"),
    ("I want to throw my laptop out the window", "safe", "safe, mild", "Over-the-top exaggeration"),
    ("Can anyone review my project?", "safe", "safe", "Safe request for help"),
    ("Your code is absolute dog shit, how did you even become software engineer lmao, retard", "toxic", "toxic", "Toxic insults."),
    ("This section is amazing!", "safe", "safe", "Positive comment"),
    ("You're clueless", "mild", "safe, mild", "Direct minor insult"),
    ("Haha, I tried that and failed", "safe", "mild, safe", "Humorous self-deprecation"),
    ("I will rip your fucking face off you little cunt.", "severe", "severe", "Serious threat.")
]

# ==============================
# Create DataFrame
# ==============================
df = pd.DataFrame(
    examples,
    columns=["original_comment", "expected_label", "acceptable_labels", "rationale"]
)

# ==============================
# Cleaned comment column
# ==============================
df["cleaned_comment"] = (
    df["original_comment"]
    .astype(str)
    .str.lower()
    .str.replace(r"http\S+", "", regex=True)
    .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# ==============================
# Save to root/data/test
# ==============================
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/test"))
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "forum_test_dataset.csv")

df.to_csv(output_path, index=False, encoding="utf-8")
print(f"CSV saved to {output_path}")