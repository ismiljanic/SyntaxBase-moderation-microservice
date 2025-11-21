import pandas as pd

def count_features(text: str):
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "num_uppercase": sum(1 for c in text if c.isupper()),
        "num_exclamation": text.count("!"),
        "num_question": text.count("?"),
        "has_swear": int(any(bad in text.lower() for bad in ["fuck","shit","idiot","stupid","dumb"]))
    }

def make_numeric_features_list(texts):
    import numpy as np
    rows = [list(count_features(t).values()) for t in texts]
    return np.array(rows)
