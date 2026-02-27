# 📰 Fake News Detection System

This project is a Machine Learning based Fake News Detection system built using Python and Scikit-learn.

## 📌 Project Overview
The model classifies news text as **Fake** or **Real** using Natural Language Processing (NLP) techniques.

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Joblib (Model Saving)

## 📂 Project Structure

fake_news_project/
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf.pkl
│
├── prepare_data.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md

## 🚀 How It Works

1. Data preprocessing is done using `prepare_data.py`
2. The model is trained using `train_model.py`
3. The trained model is saved as `.pkl` files
4. `predict.py` loads the model and predicts whether news is Fake or Real

## ▶️ How to Run

```bash
python train_model.py
python predict.py
```

## 📊 Model Details

- Feature Extraction: TF-IDF Vectorizer
- Classifier: Logistic Regression
- Output: Fake/Real Prediction with Probability

## 📈 Future Improvements

- Deploy using Streamlit
- Use Deep Learning models (LSTM/BERT)
- Add real-time news API integration

---

👨‍💻 Developed by Subham Kumar Swain