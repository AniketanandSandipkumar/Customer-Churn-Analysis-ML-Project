# 🏗️ System Architecture

The Customer Churn Prediction system follows a modular Machine Learning pipeline.

---

## Flow

User Input (Streamlit UI)  
        ↓  
Data Preprocessing  
        ↓  
Feature Engineering  
        ↓  
Trained Machine Learning Model (XGBoost)  
        ↓  
Prediction Output  
        ↓  
Explainability (SHAP)

---

## Components

### 1. Streamlit UI
Collects user input and displays prediction results.

### 2. Preprocessing Module
Handles feature transformations and ensures consistency.

### 3. Model Layer
Trained ML model used for prediction.

### 4. Explainability Layer
SHAP is used to interpret model predictions.

---

## Design Principles

- Modular architecture  
- Reusable components  
- Clear separation of concerns  
- Production-ready structure  