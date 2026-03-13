import pandas as pd
import re

# -----------------------------
# Load datasets
# -----------------------------
fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

# -----------------------------
# Add labels
# -----------------------------
fake["label"] = 0
true["label"] = 1

# -----------------------------
# Combine datasets
# -----------------------------
df = pd.concat([fake, true], ignore_index=True)

# Keep only required columns
df = df[["text", "label"]]

# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""

    text = text.lower()

    # remove news sources
    text = re.sub(r'reuters|cnn|bbc|ap|guardian', '', text, flags=re.I)

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["text"] = df["text"].apply(clean_text)

# -----------------------------
# Remove empty rows
# -----------------------------
df = df[df["text"].str.strip() != ""]

# -----------------------------
# Shuffle dataset
# -----------------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# Save dataset
# -----------------------------
df.to_csv("news_dataset.csv", index=False)

print("Dataset ready!")
print("Total samples:", df.shape)