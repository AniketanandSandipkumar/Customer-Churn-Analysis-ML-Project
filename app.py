import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import datetime

from src.preprocessing import create_features

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.metric {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("models/churn_model_latest.pkl", "rb"))
clf = model.named_steps["model"]
preprocessor = model.named_steps["preprocessor"]

# -----------------------------
# TITLE
# -----------------------------
st.markdown("## 📊 Customer Churn Prediction Dashboard")
st.caption("AI-powered churn prediction with real-time explainability")

# -----------------------------
# LAYOUT
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 👤 Customer Details")

    age = st.slider("Age", 18, 80)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure (Months)", 1, 60)
    usage = st.slider("Usage Frequency", 1, 30)
    support = st.slider("Support Calls", 0, 10)

with col2:
    st.markdown("### 📦 Subscription Details")

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
# PREDICTION BUTTON
# -----------------------------
if st.button("🚀 Predict Churn"):

    data = pd.DataFrame([[age, gender, tenure, usage, support, delay,
                          subscription, contract, spend, last]],
        columns=["Age","Gender","Tenure","Usage Frequency","Support Calls",
                 "Payment Delay","Subscription Type","Contract Length",
                 "Total Spend","Last Interaction"])

    data_fe = create_features(data.copy())

    prob = model.predict_proba(data_fe)[0][1]

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
        st.metric("Confidence", f"{(prob*100):.1f}%")

    # Log
    log_prediction(data_fe, prob)

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    st.markdown("### 🔍 Prediction Explanation")

    try:
        X_transformed = preprocessor.transform(data_fe)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_transformed[0]
            ),
            show=False
        )

        st.pyplot(fig)

    except Exception as e:
        st.warning("Explainability not available.")
        st.text(str(e))

# -----------------------------
# MONITORING DASHBOARD
# -----------------------------
st.markdown("### 📈 Prediction Monitoring")

if os.path.exists("logs/predictions_log.csv"):
    df_logs = pd.read_csv("logs/predictions_log.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(df_logs.tail(), use_container_width=True)

    with col2:
        st.line_chart(df_logs["prediction"])