import shap
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

from src.preprocessing import create_features

def explain_model():

    # -----------------------------
    # Create visualization folder
    # -----------------------------
    os.makedirs("visualizations", exist_ok=True)

    # -----------------------------
    # Load trained pipeline
    # -----------------------------
    model = pickle.load(open("models/churn_model_latest.pkl", "rb"))

    # -----------------------------
    # Extract model + preprocessor
    # -----------------------------
    clf = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    # -----------------------------
    # Load sample data (same as training)
    # -----------------------------
    df = pd.read_csv("data/customer_churn_dataset.csv")

    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    df = create_features(df)

    X = df.drop("Churn", axis=1).head(50)

    # -----------------------------
    # Transform data using pipeline
    # -----------------------------
    X_transformed = preprocessor.transform(X)

    # -----------------------------
    # SHAP Explainer (Tree-based)
    # -----------------------------
    explainer = shap.TreeExplainer(clf)

    shap_values = explainer.shap_values(X_transformed)

    # -----------------------------
    # Plot SHAP summary
    # -----------------------------
    shap.summary_plot(shap_values, X_transformed, show=False)

    plt.tight_layout()
    plt.savefig("visualizations/shap_summary.png")
    plt.close()

    print("✅ SHAP explanation saved at: visualizations/shap_summary.png")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    explain_model()