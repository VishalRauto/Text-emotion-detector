import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.utils import resample

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("emotion_dataset_raw.csv")
df.columns = df.columns.str.strip()
df["Text"] = df["Text"].str.strip()
df["Emotion"] = df["Emotion"].str.strip()
df.dropna(inplace=True)

# ── Clean ─────────────────────────────────────────────────────────────────────
def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df["Text"] = df["Text"].apply(clean)

# ── Oversample minority classes ───────────────────────────────────────────────
max_count = df["Emotion"].value_counts().max()
parts = [
    resample(g, replace=True, n_samples=max_count, random_state=42)
    if len(g) < max_count else g
    for _, g in df.groupby("Emotion")
]
df = pd.concat(parts)

X_train, X_test, y_train, y_test = train_test_split(
    df["Text"], df["Emotion"], test_size=0.2, random_state=42, stratify=df["Emotion"]
)

# ── Vectorize ─────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training...")
model = LinearSVC(C=1, max_iter=2000)
model.fit(X_train_vec, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# ── Save ──────────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "vectorizer": vectorizer}, f)
print("Model saved to model.pkl")
