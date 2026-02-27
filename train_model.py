import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("news_dataset.csv")

X = df["text"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=15000,
    stop_words='english',
    ngram_range=(1,2),
    min_df=2
)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Initialize models
lr_model = LogisticRegression(max_iter=2000, class_weight='balanced')
nb_model = MultinomialNB()
svm_model = LinearSVC()

# Train models
lr_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)


models = {
    "Logistic Regression": lr_model,
    "Naive Bayes": nb_model,
    "SVM": svm_model
}

print("\n===== Cross Validation (5-Fold) =====")

for name, model in models.items():
    scores = cross_val_score(model, X_vectorized, y, cv=5, scoring='accuracy')
    print(f"{name} CV Accuracy Scores: {scores}")
    print(f"{name} Mean CV Accuracy: {scores.mean():.4f}")

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

plt.figure()

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Confusion Matrix for SVM (Best Model)
y_pred_svm = svm_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svm)
print("\n===== Confusion Matrix (SVM) =====")
print(cm)

print("\nDetails:")
print("Real predicted as Real:", cm[0][0])
print("Real predicted as Fake:", cm[0][1])
print("Fake predicted as Real:", cm[1][0])
print("Fake predicted as Fake:", cm[1][1])
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Save model
joblib.dump(svm_model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

print("Model saved successfully!")

