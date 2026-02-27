import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🧠 Advanced AI Behaviour-Based Credit Card Fraud Detection (10K Data)")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "creditcard.csv")
    df = pd.read_csv(file_path)
    df = df.sample(n=10000, random_state=42)
    return df

df = load_data()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
MODEL_PATH = "rf_model_10k.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙ Transaction Controls")

amount_input = st.sidebar.number_input(
    "💰 Transaction Amount", 0.0, 10000.0, 500.0
)

time_input = st.sidebar.number_input(
    "⏱ Transaction Time", 0.0, float(df["Time"].max()), 1000.0
)

behaviour_type = st.sidebar.selectbox(
    "🧠 Behaviour Pattern",
    ["Normal Behaviour", "Suspicious Behaviour", "Highly Anomalous"]
)

run_button = st.sidebar.button("🚀 Run Analysis")

# --------------------------------------------------
# PCA GENERATION (REAL DATA BASED)
# --------------------------------------------------
def generate_pca(behaviour_type):
    if behaviour_type == "Normal Behaviour":
        sample = df[df["Class"] == 0].sample(1)
    else:
        sample = df[df["Class"] == 1].sample(1)

    return sample.drop("Class", axis=1).iloc[0][1:-1].values

# --------------------------------------------------
# RUN ANALYSIS
# --------------------------------------------------
if run_button:

    pca_values = generate_pca(behaviour_type)

    input_data = pd.DataFrame(
        [[time_input] + list(pca_values) + [amount_input]],
        columns=df.drop("Class", axis=1).columns
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_score = round(probability * 100, 2)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎯 Risk Score", f"{risk_score}/100")

    with col2:
        if prediction == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ Legit Transaction")

    st.markdown("---")

    # --------------------------------------------------
    # RISK GAUGE
    # --------------------------------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Fraud Risk Meter"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # EXPLANATION SECTION
    # --------------------------------------------------
    st.markdown("### 🔍 Model Explanation")

    explanation = []

    if risk_score > 75:
        explanation.append("Extremely high anomaly patterns detected in behaviour features.")
    elif risk_score > 50:
        explanation.append("Multiple suspicious behavioural indicators found.")
    else:
        explanation.append("Behaviour falls within normal transaction range.")

    if amount_input > df["Amount"].quantile(0.95):
        explanation.append("Transaction amount is unusually high compared to historical data.")

    for e in explanation:
        st.write("•", e)

    st.markdown("---")

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------
    st.subheader("📊 Top 10 Feature Importance (Model Insight)")

    feature_importance = model.feature_importances_
    feature_names = df.drop("Class", axis=1).columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Influencing Fraud Detection",
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    **Explanation:**  
    The Random Forest model assigns importance scores to each feature based on how much they contribute 
    to reducing classification error. Higher importance means the feature plays a stronger role 
    in detecting fraudulent behaviour.
    """)

else:
    st.info("👈 Configure transaction details and click 'Run Analysis'")