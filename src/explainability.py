import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import shap

# Create folder
os.makedirs("visualizations", exist_ok=True)

# Load pipeline
pipeline = pickle.load(open("models/churn_model_latest.pkl", "rb"))

# Split pipeline
model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

def explain_model():

    # Sample input
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
        "Last Interaction": 10
    }])

    # Feature Engineering
    sample["Usage_Intensity"] = sample["Usage Frequency"] / (sample["Tenure"] + 1)
    sample["Spend_per_Tenure"] = sample["Total Spend"] / (sample["Tenure"] + 1)

    # Transform data
    X_processed = preprocessor.transform(sample)

    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_processed)

    # Plot
    shap.summary_plot(shap_values, X_processed, show=False)

    plt.savefig("visualizations/shap_feature_importance.png")
    plt.close()

    print("SHAP visualization saved!")

if __name__ == "__main__":
    explain_model()
