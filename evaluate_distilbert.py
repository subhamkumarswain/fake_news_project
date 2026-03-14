import pandas as pd
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# -----------------------------
# Load model
# -----------------------------
model_path = "./distilbert_fake_news_model"

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

model.eval()

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("news_dataset.csv")

# Add short real news (same as training)
short_real = [
    "United Nations announces new climate initiative.",
    "Government approves new education policy.",
    "Central bank holds interest rates steady.",
    "Health ministry launches vaccination campaign.",
    "City council approves new park development.",
    "New transportation law passed by parliament.",
    "Scientists report progress on renewable energy.",
    "Weather department predicts heavy rainfall tomorrow.",
    "Economic growth rises in latest quarterly report.",
    "International summit focuses on climate change."
]

data = pd.concat([data, pd.DataFrame({"text": short_real, "label": 1})], ignore_index=True)

# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'reuters|cnn|bbc|ap|guardian', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data["text"] = data["text"].apply(clean_text)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# -----------------------------
# Prediction
# -----------------------------
predictions = []
probabilities = []

for text in X_test:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()

    probabilities.append(real_prob)

    if fake_prob > real_prob:
        predictions.append(0)
    else:
        predictions.append(1)

# -----------------------------
# Evaluation metrics
# -----------------------------
print("\n===== DistilBERT Evaluation =====")

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, predictions)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - DistilBERT")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"DistilBERT (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - DistilBERT")
plt.legend()
plt.show()