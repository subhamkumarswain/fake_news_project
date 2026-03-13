import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

# -----------------------------
# Load dataset and add short real headlines
# -----------------------------
df = pd.read_csv("news_dataset.csv")

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
df = pd.concat([df, pd.DataFrame({"text": short_real, "label": 1})], ignore_index=True)

# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'reuters|cnn|bbc|ap|guardian', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

# -----------------------------
# Features & labels
# -----------------------------
X = df["text"]
y = df["label"]

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.95,
    min_df=2,
    ngram_range=(1,2),
    sublinear_tf=True
)
X_vectorized = vectorizer.fit_transform(X)

# -----------------------------
# Initialize models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "Naive Bayes": MultinomialNB(),
    "SVM (Calibrated)": CalibratedClassifierCV(
        LinearSVC(class_weight={0:1,1:1.2}, random_state=42),cv=3
    )
}

# -----------------------------
# Cross-validation for all models
# -----------------------------
print("\n===== 5-Fold Cross Validation =====")
for name, model in models.items():
    scores = cross_val_score(model, X_vectorized, y, cv=5, scoring='accuracy')
    print(f"{name} CV Accuracy Scores: {scores}")
    print(f"{name} Mean CV Accuracy: {scores.mean():.4f}")

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train all models
# -----------------------------
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} training complete!")

# -----------------------------
# Evaluate all models
# -----------------------------
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    print(f"\n===== {name} Evaluation =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion matrix for SVM (main deployment model)
# -----------------------------
svm_model = trained_models["SVM (Calibrated)"]
y_pred_svm = svm_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svm)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - SVM (Calibrated)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# -----------------------------
# Combined ROC curve for all models
# -----------------------------
plt.figure()
for name, model in trained_models.items():
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:,1]
    else:
        y_scores = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - All Models")
plt.legend()
plt.show()

# -----------------------------
# Save final SVM model & vectorizer
# -----------------------------
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("SVM model and vectorizer saved successfully!")