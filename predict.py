import joblib
import re

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# -----------------------------
# Clean text (same as training)
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r'reuters|cnn|bbc|ap|guardian', '', text, flags=re.I)

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


print("Fake News Detection (type 'exit' to quit)")

while True:

    news = input("\nEnter news text: ")

    if news.lower() == "exit":
        break

    cleaned = clean_text(news)

    vector = vectorizer.transform([cleaned])

    pred = model.predict(vector)[0]

    probs = model.predict_proba(vector)[0]

    confidence = probs[pred]

    label = "FAKE" if pred == 0 else "REAL"

    print("\nPrediction:", label)
    print("Confidence:", round(confidence * 100, 2), "%")