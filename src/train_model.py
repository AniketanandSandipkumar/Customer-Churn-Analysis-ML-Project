import pandas as pd
import pickle
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

from src.preprocessing import create_features

# -----------------------------
# Create required folders
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# -----------------------------
# Load and preprocess data
# -----------------------------
df = pd.read_csv("data/customer_churn_dataset.csv")

if "CustomerID" in df.columns:
    df = df.drop("CustomerID", axis=1)

df = create_features(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------------
# Column separation
# -----------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# -----------------------------
# Preprocessing pipeline
# -----------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# -----------------------------
# Models
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = {}

best_model = None
best_score = 0
best_model_name = ""

# -----------------------------
# Train and evaluate models
# -----------------------------
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

    # Select best model
    if acc > best_score:
        best_score = acc
        best_model = pipeline
        best_model_name = name

# -----------------------------
# Save best model
# -----------------------------
model_path = "models/churn_model_latest.pkl"

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

# -----------------------------
# Save metrics report
# -----------------------------
with open("reports/metrics.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n\n")
    for model_name, metrics in results.items():
        f.write(f"{model_name}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n")

# -----------------------------
# Feature Importance (only for tree-based models)
# -----------------------------
try:
    clf = best_model.named_steps["model"]
    preproc = best_model.named_steps["preprocessor"]

    if hasattr(clf, "feature_importances_"):
        import matplotlib.pyplot as plt

        importances = clf.feature_importances_
        feature_names = preproc.get_feature_names_out()

        # Sort top 10 features
        indices = importances.argsort()[-10:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), feature_names[indices])
        plt.title(f"Top Feature Importances ({best_model_name})")
        plt.xlabel("Importance")

        plt.tight_layout()
        plt.savefig("visualizations/feature_importance.png")
        plt.close()

except Exception as e:
    print("⚠️ Feature importance skipped:", e)

# -----------------------------
# Final Output
# -----------------------------
print("✅ Training Complete")
print(f"🏆 Best Model: {best_model_name}")
print(f"📊 Best Accuracy: {best_score:.4f}")
print(f"💾 Model saved at: {model_path}")