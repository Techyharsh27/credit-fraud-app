import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Fraud Detection", layout="wide")

st.title("🧠 Advanced AI Behaviour-Based Credit Card Fraud Detection (10K Data)")

# =========================
# MODEL TRAINING FUNCTION
# =========================

@st.cache_resource
def train_model():
    df = pd.read_csv(
        "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    )

    # Use only 10k rows
    df = df.sample(n=10000, random_state=42)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    return model


model = train_model()

# =========================
# SIDEBAR INPUT
# =========================

st.sidebar.header("⚙ Transaction Controls")

amount = st.sidebar.number_input("💰 Transaction Amount", 0.0, 50000.0, 1000.0)
time = st.sidebar.number_input("⏱ Transaction Time", 0.0, 200000.0, 50000.0)

behaviour = st.sidebar.selectbox(
    "🧠 Behaviour Pattern",
    ["Normal", "Suspicious", "Highly Anomalous"]
)

run_button = st.sidebar.button("🚀 Run Analysis")

# =========================
# PREDICTION LOGIC
# =========================

if run_button:

    # Create dummy input (Time + V1-V28 + Amount)
    input_data = [time] + [0]*28 + [amount]
    input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Risk score calculation
    risk_score = probability * 100

    if behaviour == "Suspicious":
        risk_score += 15
    elif behaviour == "Highly Anomalous":
        risk_score += 30

    risk_score = min(risk_score, 100)

    st.subheader("🎯 Risk Score")
    st.metric("Fraud Risk Score", f"{risk_score:.1f}/100")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.progress(int(risk_score))

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("Model trained on 10,000 sampled transactions. Auto-trained at startup.")