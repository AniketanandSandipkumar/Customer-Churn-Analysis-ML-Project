import pickle
import pandas as pd

model = pickle.load(open("models/churn_model_latest.pkl", "rb"))

sample = pd.DataFrame([[30,"Male",10,15,2,5,"Basic","Monthly",500,10]],
columns=["Age","Gender","Tenure","Usage Frequency","Support Calls",
         "Payment Delay","Subscription Type","Contract Length",
         "Total Spend","Last Interaction"])

pred = model.predict(sample)
print("Prediction:", pred)