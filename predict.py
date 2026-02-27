import joblib

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

text = input("\nEnter news text:\n")

text_vectorized = vectorizer.transform([text])

prob = model.predict_proba(text_vectorized)[0]
prediction = model.predict(text_vectorized)[0]

print("\n===== Prediction Result =====")
print(f"Fake News Probability: {prob[0]*100:.2f}%")
print(f"Real News Probability: {prob[1]*100:.2f}%")

if prediction == 0:
    print("\nFinal Prediction: FAKE NEWS")
else:
    print("\nFinal Prediction: REAL NEWS")


print("classes:",model.classes_)