import pandas as pd

fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)

df = df[["text", "label"]]
df = df.sample(frac=1, random_state=42)

df.to_csv("news_dataset.csv", index=False)

print("Dataset ready!")
print("Total samples:", df.shape)