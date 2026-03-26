"""
Downloads GoEmotions dataset (58k Reddit comments) from HuggingFace,
maps its 27 emotions to our 8 labels, merges with existing dataset,
and saves as emotion_dataset_merged.csv
"""
from datasets import load_dataset
import pandas as pd

print("Downloading GoEmotions...")
ds = load_dataset("go_emotions", "simplified")

# GoEmotions simplified has these labels:
label_names = ds["train"].features["labels"].feature.names
print("GoEmotions labels:", label_names)

# Map GoEmotions 28 labels -> our 8
mapping = {
    "admiration": "joy", "amusement": "joy", "approval": "joy",
    "caring": "joy", "desire": "joy", "excitement": "joy",
    "gratitude": "joy", "joy": "joy", "love": "joy",
    "optimism": "joy", "pride": "joy", "relief": "joy",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "sadness": "sadness", "grief": "sadness", "remorse": "sadness",
    "embarrassment": "shame", "confusion": "neutral",
    "curiosity": "surprise", "realization": "surprise", "surprise": "surprise",
    "neutral": "neutral",
}

rows = []
for split in ["train", "validation", "test"]:
    for item in ds[split]:
        if not item["labels"]:
            continue
        # take first label
        lname = label_names[item["labels"][0]]
        mapped = mapping.get(lname)
        if mapped:
            rows.append({"Emotion": mapped, "Text": item["text"]})

go_df = pd.DataFrame(rows)
print("GoEmotions mapped:", len(go_df))
print(go_df["Emotion"].value_counts())

# Load existing dataset
orig = pd.read_csv("emotion_dataset_raw.csv")
orig.columns = orig.columns.str.strip()
orig["Text"] = orig["Text"].str.strip()
orig["Emotion"] = orig["Emotion"].str.strip()
orig.dropna(inplace=True)

# Merge
merged = pd.concat([orig, go_df], ignore_index=True)
merged.drop_duplicates(subset="Text", inplace=True)
merged.to_csv("emotion_dataset_merged.csv", index=False)
print(f"\nMerged dataset: {len(merged)} rows")
print(merged["Emotion"].value_counts())
print("Saved to emotion_dataset_merged.csv")
