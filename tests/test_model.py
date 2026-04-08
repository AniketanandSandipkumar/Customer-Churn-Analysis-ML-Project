import pickle
import pandas as pd

def test_model_prediction():
    model = pickle.load(open("models/churn_model_latest.pkl", "rb"))

    sample = pd.DataFrame([{
        "Age": 30,
        "Gender": "Male",
        "Tenure": 12,
        "Usage Frequency": 10,
        "Support Calls": 2,
        "Payment Delay": 5,
        "Subscription Type": "Standard",
        "Contract Length": "Monthly",
        "Total Spend": 500,
        "Last Interaction": 10,
        "Usage_Intensity": 10/(12+1),
        "Spend_per_Tenure": 500/(12+1)
    }])

    pred = model.predict(sample)

    assert pred is not None