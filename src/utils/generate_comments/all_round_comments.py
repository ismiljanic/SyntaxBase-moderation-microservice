import pandas as pd
import os

examples = [
    ("I'm going to kill you.", "severe", "toxic, severe", "Direct first-person death threat toward addressee."),
    ("I will punch him next time I see him.", "mild", "mild, toxic", "Threat of violence but less explicit lethal intent."),
    ("I want to chop your head off.", "severe", "toxic, severe", "Graphic violent threat."),
    ("I want to kill myself.", "mild", "mild,safe", "Self-harm (non-directed), mild or safe depending on context."),
    ("What are you talking about, you dumb idiot?", "toxic", "toxic,mild", "Insult / harassment, not violent."),
    ("She looks like such a retard.", "toxic", "toxic,severe", "Highly offensive slur directed at protected characteristic — severe."),
    ("Haha I’ll kill you (just kidding bro)", "mild", "mild,safe", "Threat phrased as joke — ambiguous/contextual."),
    ("If you don't stop, I'll make you regret it.", "toxic", "toxic, mild", "Conditional threat — threatening but imprecise."),
    ("I'm going to murder your family.", "severe", "toxic, severe", "Threat to third parties/others — high severity."),
    ("c u n t", "toxic", "toxic", "Explicit slur/insult despite spaced obfuscation; tests tokenization robustness."),
    ("f*** you", "toxic", "toxic", "Explicit insult with masking characters."),
    ("I will find you and hurt you.", "toxic", "toxic, mild", "Targeted violent threat with intent."),
    ("I want to study a killer's mindset.", "safe", "safe,mild", "Non-threatening, academic intent — tests false positive for violent terms."),
    ("I would never kill you.", "safe", "safe", "Negated threat — tests negation handling."),
    ("Go back to where you came from.", "mild", "mild, toxic", "Xenophobic exclusion — harassment towards protected class."),
    ("You’re so stupid lol", "mild", "mild, toxic", "Mild insult / low severity."),
    ("What are you thinking you little faggot i hope your whole family burns alive. Fuck you and your little forum SyntaxBase jews.", "severe", "severe,toxic", "Serious racial and gender slurs."),
    ("He said 'I'll slit her throat' in the chat.", "safe", "safe, mild", "Quoted speech / reporting — distinguishes quoting vs endorsing."),
    ("I want to stab a cop", "severe", "toxic, severe", "Violent intent toward a protected profession — high severity."),
    ("I will beat you up when you come here.", "toxic", "toxic, mild", "Physical threat, contextual severity moderate–high."),
    ("You fucking piece of garbage", "toxic", "toxic, severe", "Strong insult but not violent."),
    ("Drop dead.", "toxic", "toxic", "Wishing death, violent-directed phrase."),
    ("He should be locked up for life.", "safe", "safe", "Opinion about punishment / criminal justice — not necessarily toxic."),
    ("fuckkerrr", "toxic", "toxic, mild", "Obfuscation via elongation; tests subword robustness."),
    ("I’m going to burn that place down (hypothetical).", "mild", "safe,mild", "Property damage threat/homicidal intent depending on context."),
    ("I love you man, I'd never hurt you.", "safe", "safe", "Positive sentiment + negation; tests polarity & coreference."),
    ("If I were you I'd quit — you suck.", "mild", "mild", "Advice + mild insult."),
    ("You are a disgusting [protected slur]", "mild", "mild, toxic", "Targeted protected-class slur — highest severity."),
    ("Can you pass me the knife?", "safe", "safe", "Innocuous request containing an instrument word."),
    ("I've dreamed of killing him many times.", "mild", "toxic, mild", "Expresses violent intent/fantasy — likely toxic/severe depending on policy."),
    ("lol 😂 kill me now, this homework", "mild", "mild,safe", "Hyperbolic self-harm as joke — should be mild/non-toxic but sensitive."),
    ("I will rip your fucking face off you little cunt.", "severe", "severe", "Serious threat.")
]

df = pd.DataFrame(examples, columns=["original_comment", "expected_label", "acceptable_labels", "rationale"])
df["cleaned_comment"] = (
    df["original_comment"]
    .astype(str)
    .str.lower()
    .str.replace(r"http\S+", "", regex=True)
    .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/test"))
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "all_round_comments_with_labels.csv")

df.to_csv(output_path, index=False, encoding="utf-8")
print(f"CSV saved to {output_path}")