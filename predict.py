import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

print("\n========== Fake News Detection System ==========")

while True:
    text = input("\nEnter news text (or type 'exit' to quit):\n")

    if text.lower() == "exit":
        print("\nExiting system... Goodbye 👋")
        break

    if len(text.strip()) == 0:
        print("Please enter valid text.")
        continue

    # Vectorize
    text_vectorized = vectorizer.transform([text])

    # Predict
    prediction = model.predict(text_vectorized)[0]

    # For LinearSVC (no predict_proba)
    decision_score = model.decision_function(text_vectorized)[0]

    # Convert decision score to confidence percentage
    confidence = abs(decision_score)
    confidence_percent = min(confidence * 10, 100)  # scaled for readability

    print("\n========== Prediction Result ==========")

    if prediction == 0:
        print("Final Prediction: FAKE NEWS ❌")
    else:
        print("Final Prediction: REAL NEWS ✅")

    print(f"Confidence Score: {confidence_percent:.2f}%")
    print(f"Model Used: {type(model).__name__}")