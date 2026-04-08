import streamlit as st
import pickle
import pandas as pd
from src.validation import validate_input
from src.preprocessing import create_features

model = pickle.load(open("models/churn_model_latest.pkl", "rb"))

st.title("📊 Customer Churn Prediction")

age = st.slider("Age", 18, 80)
gender = st.selectbox("Gender", ["Male","Female"])
tenure = st.slider("Tenure", 1, 60)
usage = st.slider("Usage Frequency", 1, 30)
support = st.slider("Support Calls", 0, 10)
delay = st.slider("Payment Delay", 0, 30)
subscription = st.selectbox("Subscription Type", ["Basic","Standard","Premium"])
contract = st.selectbox("Contract Length", ["Monthly","Quarterly","Annual"])
spend = st.number_input("Total Spend")
last = st.slider("Last Interaction", 1, 30)

if st.button("Predict"):
    data = pd.DataFrame([[age,gender,tenure,usage,support,delay,
                          subscription,contract,spend,last]],
    columns=["Age","Gender","Tenure","Usage Frequency","Support Calls",
             "Payment Delay","Subscription Type","Contract Length",
             "Total Spend","Last Interaction"])

    valid, msg = validate_input({"Age":age,"Total Spend":spend})

    if not valid:
        st.error(msg)
    else:
        # ✅ APPLY FEATURE ENGINEERING HERE
        data = create_features(data)

        prob = model.predict_proba(data)[0][1]
        st.write("Churn Probability:", round(prob, 2))
