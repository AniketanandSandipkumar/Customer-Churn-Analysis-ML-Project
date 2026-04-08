import pandas as pd
import pickle
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

# Load data
df = pd.read_csv("data/customer_churn_dataset.csv")
df = df.drop("CustomerID", axis=1)

df = create_features(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

import matplotlib.pyplot as plt
import os

os.makedirs("visualizations", exist_ok=True)

# Example: feature importance (for tree models)
try:
    importances = model.named_steps["model"].feature_importances_
    features = model.named_steps["preprocessor"].get_feature_names_out()

    plt.figure(figsize=(8,5))
    plt.barh(features[:10], importances[:10])
    plt.title("Top Feature Importances")
    plt.savefig("visualizations/feature_importance.png")
    plt.close()
except:
    pass

# Save best model (use XGBoost)
version = datetime.now().strftime("%Y%m%d_%H%M")
model_path = f"models/churn_model_latest.pkl"

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)


# Save latest model (IMPORTANT)
with open("models/churn_model_latest.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Save metrics
with open("reports/metrics.txt", "w") as f:
    for m, vals in results.items():
        f.write(f"{m}\n{vals}\n\n")

print("✅ Training Complete")