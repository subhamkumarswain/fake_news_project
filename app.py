import streamlit as st
import joblib
import re
import plotly.graph_objects as go
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# -----------------------------
# Load SVM model
# -----------------------------
svm_model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# -----------------------------
# Load DistilBERT model
# -----------------------------
bert_model_path = "distilbert_fake_news_model"

tokenizer = DistilBertTokenizer.from_pretrained(bert_model_path)
bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)

# -----------------------------
# Clean text function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'reuters|cnn|bbc|ap|guardian', '', text, flags=re.I)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Confidence Gauge
# -----------------------------
def show_gauge(confidence):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Prediction Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")

st.info(
"This application detects fake news using two machine learning approaches: "
"SVM and DistilBERT."
)

# Model selector
model_choice = st.selectbox(
"Select Model",
["SVM (Baseline ML)", "DistilBERT (Transformer)"]
)

news_input = st.text_area("Enter News Article Text")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Analyze News"):

    if news_input.strip() == "":
        st.warning("Please enter some news text.")

    else:

        cleaned = clean_text(news_input)

        # -----------------------------
        # SVM MODEL
        # -----------------------------
        if model_choice == "SVM (Baseline ML)":

            vector = vectorizer.transform([cleaned])

            pred = svm_model.predict(vector)[0]
            probs = svm_model.predict_proba(vector)[0]

            confidence = probs[pred]

            label = "FAKE" if pred == 0 else "REAL"

            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {confidence*100:.2f}%")

            show_gauge(confidence)

            # Important words
            st.markdown("### 🔎 Important Words")

            feature_names = vectorizer.get_feature_names_out()
            tfidf_values = vector.toarray()[0]

            top_indices = tfidf_values.argsort()[-10:][::-1]

            for i in top_indices:
                if tfidf_values[i] > 0:
                    st.write(feature_names[i])

        # -----------------------------
        # DISTILBERT MODEL
        # -----------------------------
        else:

            inputs = tokenizer(
                news_input,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = bert_model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)[0]

            fake_prob = probs[0].item()
            real_prob = probs[1].item()

            pred = 1 if real_prob > fake_prob else 0
            confidence = max(fake_prob, real_prob)

            label = "REAL" if pred == 1 else "FAKE"

            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {confidence*100:.2f}%")

            show_gauge(confidence)

st.markdown("---")

st.markdown("""
### About this Project

This Fake News Detection system compares two machine learning approaches:

• SVM with TF-IDF features  
• DistilBERT transformer model

The system analyzes linguistic patterns and contextual understanding to detect misinformation.
""")