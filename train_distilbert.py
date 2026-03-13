import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch.nn as nn


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("news_dataset.csv")

# sample for faster training
df = df.sample(6000, random_state=42)

texts = df["text"].tolist()
labels = df["label"].tolist()


# -----------------------------
# Train test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding=True,
    max_length=256
)

test_encodings = tokenizer(
    X_test,
    truncation=True,
    padding=True,
    max_length=256
)


# -----------------------------
# Dataset class
# -----------------------------
class NewsDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_encodings, y_train)
test_dataset = NewsDataset(test_encodings, y_test)


# -----------------------------
# Load model
# -----------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# -----------------------------
# Freeze DistilBERT layers
# -----------------------------
for param in model.distilbert.parameters():
    param.requires_grad = False


# -----------------------------
# Class weights (fix bias)
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = torch.tensor(class_weights, dtype=torch.float)


# -----------------------------
# Custom Trainer
# -----------------------------
class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):

        labels = inputs.get("labels")

        outputs = model(**inputs)

        logits = outputs.get("logits")

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=3,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    logging_dir="./logs",

)


# -----------------------------
# Trainer
# -----------------------------
trainer = CustomTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

)


# -----------------------------
# Train model
# -----------------------------
trainer.train()


# -----------------------------
# Save model
# -----------------------------
trainer.save_model("distilbert_fake_news_model")

tokenizer.save_pretrained("distilbert_fake_news_model")


print("Training completed!")
print("Model saved successfully!")