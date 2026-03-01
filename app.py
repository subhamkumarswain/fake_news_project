import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Load model
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

# Title
st.title("📰 Fake News Detection System")
st.markdown("Detect whether a news article is **Fake** or **Real** using Machine Learning (SVM + TF-IDF).")

st.markdown("---")

# Sidebar Info
st.sidebar.header("📊 Model Information")
st.sidebar.write("Algorithm: LinearSVC")
st.sidebar.write("Vectorizer: TF-IDF")
st.sidebar.write("Cross Validation Accuracy: 99.5%")

st.sidebar.markdown("---")
st.sidebar.write("Developed by Subham Kumar Swain")

# User Input
user_input = st.text_area("Enter News Text Below")

if st.button("🔍 Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        text_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(text_vectorized)[0]
        decision_score = model.decision_function(text_vectorized)[0]

        confidence = min(abs(decision_score) * 10, 100)

        st.markdown("### 📌 Prediction Result")

        if prediction == 0:
            st.error("❌ FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

        st.write(f"Confidence Score: {confidence:.2f}%")

st.markdown("---")
st.markdown("⚠️ **Note:** This model detects writing style patterns learned from dataset. It does not verify factual truth.")
