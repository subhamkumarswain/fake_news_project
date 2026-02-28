# 📰 Fake News Detection using Machine Learning

## 📌 Project Overview

Fake news spreads misinformation rapidly through digital platforms.  
This project builds a Machine Learning-based Fake News Detection system using Natural Language Processing (NLP) techniques to classify news articles as **Real (0)** or **Fake (1)**.

The system compares multiple ML models and evaluates them using accuracy, classification metrics, and ROC-AUC score.

---

## 📊 Dataset

- Total Samples: **44,898**
- Columns: `text`, `label`
- Label:
  - `0` → Fake News
  - `1` → Real News

The dataset was preprocessed and combined into a single CSV file before training.

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- TF-IDF Vectorizer
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (LinearSVC)
- Matplotlib
- Joblib

---

## ⚙️ Project Workflow

1. Load dataset using Pandas  
2. Text feature extraction using **TF-IDF Vectorization**
   - max_features = 15000  
   - ngram_range = (1,2)  
   - stop_words = 'english'  
3. Train-Test Split (80% Training, 20% Testing)  
4. Train multiple models:
   - Logistic Regression  
   - Naive Bayes  
   - Linear SVM  
5. Evaluate models using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  
6. Plot ROC Curve and compute AUC score  
7. Save best performing model using Joblib  

---

## 📈 Model Comparison

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 98.66% |
| Naive Bayes | 94.89% |
| Linear SVM | 99.38% |

✅ **Best Performing Model: Linear SVM**

---

## 📊 ROC Curve Analysis

ROC Curve was plotted for all models using:

- False Positive Rate (FPR)
- True Positive Rate (TPR)
- Area Under Curve (AUC)

Higher AUC indicates better classification performance.

---

## 💾 Model Saving

The trained model and TF-IDF vectorizer are saved using `joblib`:

- `models/fake_news_model.pkl`
- `models/tfidf.pkl`

This allows direct loading for prediction without retraining.

---

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
python prepare_data.py
python train_model.py
python predict.py
```
---

👨‍💻 Developed by Subham Kumar Swain
