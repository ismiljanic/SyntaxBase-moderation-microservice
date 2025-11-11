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
    ("I will rip your fucking face off you little cunt.", "severe", "severe", "Serious threat."),
    ("Can somebody point me to a good beginner React tutorial?", "safe", "safe", "Help-seeking, neutral"),
    ("Great explanation — cleared up my confusion, thanks!", "safe", "safe", "Positive feedback"),
    ("This answer is kind of useless", "mild", "mild, safe", "Dismissive but not abusive"),
    ("Try reading the docs before spamming the forum", "mild", "mild, safe", "Mild admonishment"),
    ("What an elegant solution, well done!", "safe", "safe", "Complimentary technical praise"),
    ("You should just quit programming", "mild", "toxic, mild", "Derogatory suggestion"),
    ("This is garbage, what a waste of time", "mild", "mild, toxic", "Strong negative opinion"),
    ("Stop being dense and learn how scope works", "mild", "safe, mild", "Insulting, demeaning language"),
    ("I almost cried laughing at this terrible example", "mild", "mild, safe", "Exaggerated critique"),
    ("Can anyone share test cases for edge conditions?", "safe", "safe", "Technical request"),
    ("Your PR broke the build, fix it you dumb fucking idiot", "toxic", "toxic", "Direct insult + blame"),
    ("That solution is brilliant — saved my weekend", "safe", "safe", "Positive praise"),
    ("Please don't post spoilers without a warning", "safe", "safe, mild", "Polite request with mild reproach"),
    ("You clearly copy-pasted this from a blog", "safe", "mild, safe", "Accusation, minor hostility"),
    ("I can't believe someone with your experience wrote that", "mild", "mild, safe", "Patronizing and dismissive"),
    ("This thread is a dumpster fire for retards", "toxic", "toxic, severe", "Derogatory metaphor"),
    ("Thanks — that edge case saved me a headache", "safe", "safe", "Grateful response"),
    ("Your comment is pointless and stupid just like you", "toxic", "toxic, mild", "Direct insult"),
    ("I'm going to find out where you live and make you regret this you fucking retard", "severe", "severe, toxic", "Explicit threat"),
    ("Not helpful at all, move on", "mild", "mild, safe", "Unhelpful dismissal"),
    ("That's the dumbest thing I've seen all week", "mild", "toxic, mild", "Harsh criticism bordering on insult"),
    ("Could you explain why this works for big-O?", "safe", "safe", "Technical clarification request"),
    ("You're a dumbass clown, stop pretending to know stuff", "toxic", "toxic, mild", "Insulting ridicule"),
    ("This fix saved production — you're a lifesaver", "safe", "safe", "Gratitude and praise"),
    ("I hope someone reports you for spamming", "safe", "safe, mild", "Hostile wish directed at user"),
    ("Why would you even use that library? Terrible choice", "mild", "mild, safe", "Opinionated critique"),
    ("Shut up, nobody asked for your opinion shitface", "toxic", "toxic, mild", "Rude dismissal"),
    ("I'm so done with this project, thinking of quitting", "safe", "safe, mild", "Personal frustration, non-directed"),
    ("You are an absolute disgrace to developers everywhere", "mild", "toxic, mild", "Personal attack"),
    ("If you ever post this again I'll come to your city and make you pay", "toxic", "toxic", "Direct violent threat"),
    ("How can I optimize this SQL query?", "safe", "safe", "Technical question, safe"),
    ("Nice tutorial, very clean and structured", "safe", "safe", "Positive feedback"),
    ("I'm lost, can someone explain recursion again?", "safe", "safe", "Help request"),
    ("You have no idea what you're talking about dumbass", "toxic", "toxic, mild", "Insulting tone"),
    ("This update broke my app!", "safe", "safe, mild", "Complaint, mild frustration"),
    ("Why is everyone ignoring my question?", "safe", "safe, mild", "Frustration, not abusive"),
    ("Brilliant workaround, thank you!", "safe", "safe", "Praise, positive"),
    ("Who wrote this garbage code?", "mild", "mild, toxic", "Criticism, slightly rude"),
    ("Your solution doesn't even compile", "safe", "safe, mild", "Technical criticism"),
    ("Stop pretending you know C++ you suck at it. Better yet, quit programming for life loser.", "toxic", "toxic, mild", "Mocking, aggressive"),
    ("I learned more from the comments than the article", "safe", "safe", "Constructive observation"),
    ("You’re absolutely clueless about async programming", "mild", "toxic", "Direct insult"),
    ("This example is outdated but still helpful", "safe", "safe", "Balanced, neutral"),
    ("I can’t believe people upvoted this nonsense", "mild", "mild, toxic", "Dismissive, sarcastic"),
    ("Good job, this finally worked for me", "safe", "safe", "Positive feedback"),
    ("You sound like a complete idiot lmao", "toxic", "toxic", "Personal insult"),
    ("I respectfully disagree with this approach", "safe", "safe", "Polite disagreement"),
    ("Absolute legend, thank you for sharing", "safe", "safe", "Positive praise"),
    ("You're a waste of bandwidth", "mild", "toxic, mild", "Abusive insult"),
    ("This post deserves more visibility", "safe", "safe", "Encouraging comment"),
    ("Stop giving bad advice to beginners", "safe", "mild, safe", "Harsh tone but technically relevant"),
    ("I love this community!", "safe", "safe", "Positive engagement"),
    ("Your formatting is awful", "mild", "mild, safe", "Critique, mild insult"),
    ("This doesn’t make any sense, are you high or retarded?", "toxic", "toxic, mild", "Mocking language"),
    ("Please be more specific about your issue", "safe", "safe", "Constructive advice"),
    ("What a dumb question from dumb person", "toxic", "toxic, mild", "Insulting tone"),
    ("That’s not how memory allocation works, dude", "mild", "mild, safe", "Informal correction"),
    ("Appreciate your detailed explanation!", "safe", "safe", "Positive response"),
    ("You must be trolling", "mild", "mild, safe", "Dismissive assumption"),
    ("You're the reason this forum is dying", "toxic", "toxic, mild", "Aggressive blame"),
    ("Finally fixed the bug, thanks to your hint!", "safe", "safe", "Positive feedback"),
    ("Could someone review my pull request?", "safe", "safe", "Professional inquiry"),
    ("I’m so tired of this stupid framework", "mild", "mild, safe", "Negative opinion, mild"),
    ("Shut your mouth before I shut it for you fucking nigger.", "severe", "severe, toxic", "Violent threat"),
    ("Your explanation helped me understand closures", "safe", "safe", "Appreciative"),
    ("If you can't code, just quit already fcking loser", "toxic", "toxic, mild", "Insulting advice"),
    ("This looks promising, will test it later", "safe", "safe", "Neutral and positive"),
    ("Learn to spell before posting, moron", "toxic", "toxic", "Insult based on writing"),
    ("This method is deprecated, use the new API", "safe", "safe", "Technical correction"),
    ("You absolute moron, that's not how arrays work", "toxic", "toxic", "Personal insult"),
    ("I can’t stop laughing at how bad this is", "mild", "mild, toxic", "Mocking humor, not severe"),
    ("Thanks for keeping this repo updated!", "safe", "safe", "Gratitude, positive"),
    ("Seriously, read the documentation before asking dumb questions", "mild", "mild, toxic", "Slightly aggressive criticism"),
    ("Your algorithm is completely wrong, fix it", "mild", "mild, toxic", "Harsh critique, not personal"),
    ("Great job implementing that feature!", "safe", "safe", "Positive feedback"),
    ("I literally can't understand what you wrote here", "mild", "mild, safe", "Mild frustration"),
    ("Wow, this is genius! Learned a lot.", "safe", "safe", "Praise and positive engagement"),
    ("Your approach is idiotic, go back to basics", "mild", "toxic, mild", "Insulting advice"),
    ("This code snippet is useful, thanks!", "safe", "safe", "Helpful and polite"),
    ("LOL, your logic makes zero sense", "safe", "mild, toxic", "Mocking comment"),
    ("Stop spamming useless comments", "mild", "mild, toxic", "Critique of behavior"),
    ("Incredible work, keep it up!", "safe", "safe", "Encouragement and praise")
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