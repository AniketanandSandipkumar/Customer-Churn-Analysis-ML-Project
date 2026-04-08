import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix

model = pickle.load(open("models/churn_model_latest.pkl", "rb"))

df = pd.read_csv("data/customer_churn_dataset.csv")
df = df.drop("CustomerID", axis=1)

X = df.drop("Churn", axis=1)
y = df["Churn"]

pred = model.predict(X)

print(confusion_matrix(y, pred))