import streamlit as st
import pandas as pd
import requests
import os
import datetime

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "https://customer-churn-analysis-ml-project.onrender.com/predict"   

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# CUSTOM UI
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("## 📊 Customer Churn Prediction Dashboard")
st.caption("AI-powered churn prediction using deployed ML API")

st.info("⚠️ First prediction may take ~30 seconds (backend cold start)")

# -----------------------------
# INPUT UI
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure (Months)", 1, 60)
    usage = st.slider("Usage Frequency", 1, 30)
    support = st.slider("Support Calls", 0, 10)

with col2:
    delay = st.slider("Payment Delay", 0, 30)
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    spend = st.number_input("Total Spend", min_value=0.0)
    last = st.slider("Last Interaction (Days)", 1, 30)

# -----------------------------
# LOGGING
# -----------------------------
def log_prediction(data, prob):
    log_file = "logs/predictions_log.csv"
    data["prediction"] = prob
    data["timestamp"] = datetime.datetime.now()

    os.makedirs("logs", exist_ok=True)

    if not os.path.exists(log_file):
        data.to_csv(log_file, index=False)
    else:
        data.to_csv(log_file, mode='a', header=False, index=False)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict Churn"):

    input_data = {
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support,
        "Payment Delay": delay,
        "Subscription Type": subscription,
        "Contract Length": contract,
        "Total Spend": spend,
        "Last Interaction": last
    }

    try:
        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            prob = response.json()["churn_probability"]

            # -----------------------------
            # RESULT DISPLAY
            # -----------------------------
            st.markdown("### 📊 Prediction Result")

            colA, colB, colC = st.columns(3)

            with colA:
                st.metric("Churn Probability", f"{prob:.2f}")

            with colB:
                risk = "High Risk 🔴" if prob > 0.7 else "Medium 🟠" if prob > 0.4 else "Low 🟢"
                st.metric("Risk Level", risk)

            with colC:
                st.metric("Confidence", f"{prob*100:.1f}%")

            # Log
            df = pd.DataFrame([input_data])
            log_prediction(df, prob)

        else:
            st.error("❌ API Error. Please try again.")

    except Exception as e:
        st.error("⚠️ Backend not reachable (maybe sleeping). Try again in 30 seconds.")
        st.text(str(e))

# -----------------------------
# MONITORING
# -----------------------------
st.markdown("### 📈 Prediction Monitoring")

if os.path.exists("logs/predictions_log.csv"):
    df_logs = pd.read_csv("logs/predictions_log.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(df_logs.tail(), width='stretch')

    with col2:
        st.line_chart(df_logs["prediction"])
