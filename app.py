import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="AI Credit Card Fraud Detection", layout="wide")

st.title("🧠 Advanced AI Behaviour-Based Credit Card Fraud Detection (10K Data)")

# ===============================
# DATA LOADING (Render Compatible)
# ===============================

@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    df = df.sample(10000, random_state=42)  # Only 10K rows
    return df

df = load_data()

# ===============================
# MODEL TRAINING (Auto Train if Not Exists)
# ===============================

@st.cache_resource
def train_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    return model, X.columns

model, feature_columns = train_model()

# ===============================
# SIDEBAR INPUTS
# ===============================

st.sidebar.header("Transaction Controls")

amount = st.sidebar.number_input("Transaction Amount", 0.0, 50000.0, 1000.0)
time = st.sidebar.number_input("Transaction Time", 0.0, 200000.0, 10000.0)

if st.sidebar.button("🚀 Run Analysis"):

    input_data = np.zeros(len(feature_columns))
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    if "Amount" in input_df.columns:
        input_df["Amount"] = amount
    if "Time" in input_df.columns:
        input_df["Time"] = time

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    risk_score = round(probability * 100, 2)

    st.subheader("🎯 Risk Score")
    st.metric("Fraud Risk Score", f"{risk_score}/100")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction Looks Safe")

    # ===============================
    # Feature Importance
    # ===============================

    st.subheader("📊 Feature Importance")

    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(feature_importance_df.set_index("Feature"))