import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Sample data
data = {
    "text": [
        "I love this product",
        "Horrible service, will never come back",
        "Absolutely amazing experience",
        "Worst food ever",
        "Not bad, but could be better",
        "Totally loved it!",
        "I hate this app",
        "Very satisfying",
        "Disappointing result",
        "Awesome job"
    ],
    "label": ["positive", "negative", "positive", "negative", "neutral", "positive", "negative", "positive", "negative", "positive"]
}

df = pd.DataFrame(data)

# Encode labels
df["label"] = df["label"].map({"negative": 0, "neutral": 1, "positive": 2})

# Build model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200))
])

# Train
model.fit(df["text"], df["label"])

# Save model
joblib.dump(model, "sentiment_model.pkl")
