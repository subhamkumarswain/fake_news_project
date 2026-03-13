import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model_path = "./distilbert_fake_news_model"

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Set evaluation mode
model.eval()

device = torch.device("cpu")
model.to(device)

print("DistilBERT Fake News Detector Ready")
print("Type 'exit' to stop\n")

while True:

    text = input("Enter news: ")

    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Disable gradients for faster inference
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()

    print("\nFake probability:", round(fake_prob, 4))
    print("Real probability:", round(real_prob, 4))

    if fake_prob > real_prob:
        print("Prediction: Fake News\n")
    else:
        print("Prediction: Real News\n")