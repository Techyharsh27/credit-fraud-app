# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# import os
# import time

# st.set_page_config(layout="wide")

# # ---------------------------
# # DARK / LIGHT MODE
# # ---------------------------
# mode = st.sidebar.toggle("🌗 Dark Mode", value=True)

# if mode:
#     bg_color = "#0E1117"
#     text_color = "white"
# else:
#     bg_color = "white"
#     text_color = "black"

# st.markdown(f"""
# <style>
# .stApp {{
#     background-color: {bg_color};
#     color: {text_color};
# }}
# div[data-testid="metric-container"] {{
#     background-color: rgba(255,255,255,0.05);
#     padding: 20px;
#     border-radius: 12px;
#     box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
# }}
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # LOAD DATA
# # ---------------------------
# @st.cache_data
# def load_data():
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(BASE_DIR, "data", "credit_card_fraud.csv")
#     df = pd.read_csv(file_path)
#     return df

# df = load_data()

# # Rename Class column to IsFraud
# df.rename(columns={"Class": "IsFraud"}, inplace=True)

# # Create Fake Probability (since dataset doesn't have one)
# df["FraudProbability"] = np.random.uniform(0.1, 0.9, len(df))

# # ---------------------------
# # FILTERS
# # ---------------------------
# st.sidebar.header("🔎 Filters")

# amount_range = st.sidebar.slider(
#     "Select Amount Range",
#     float(df["Amount"].min()),
#     float(df["Amount"].max()),
#     (float(df["Amount"].min()), float(df["Amount"].max()))
# )

# filtered_df = df[
#     (df["Amount"] >= amount_range[0]) &
#     (df["Amount"] <= amount_range[1])
# ]

# # ---------------------------
# # ANIMATED COUNTERS
# # ---------------------------
# col1, col2, col3 = st.columns(3)

# total_tx = len(filtered_df)
# fraud_tx = filtered_df["IsFraud"].sum()
# fraud_rate = round((fraud_tx / total_tx) * 100, 2)

# def animated_counter(label, value):
#     placeholder = st.empty()
#     for i in range(0, int(value)+1, max(1, int(value)//50 + 1)):
#         placeholder.metric(label, i)
#         time.sleep(0.005)
#     placeholder.metric(label, value)

# with col1:
#     animated_counter("💳 Total Transactions", total_tx)

# with col2:
#     animated_counter("🚨 Fraud Cases", fraud_tx)

# with col3:
#     st.metric("📊 Fraud Rate (%)", fraud_rate)

# st.markdown("---")

# # ---------------------------
# # PIE CHART
# # ---------------------------
# st.subheader("Fraud vs Legit Distribution")

# pie_data = filtered_df["IsFraud"].value_counts().reset_index()
# pie_data.columns = ["FraudLabel", "Count"]
# pie_data["FraudLabel"] = pie_data["FraudLabel"].replace({0: "Legit", 1: "Fraud"})

# fig_pie = px.pie(
#     pie_data,
#     names="FraudLabel",
#     values="Count",
#     hole=0.4,
#     color="FraudLabel",
#     color_discrete_map={"Fraud": "red", "Legit": "green"}
# )

# st.plotly_chart(fig_pie, use_container_width=True)

# st.markdown("---")

# # ---------------------------
# # MODEL PERFORMANCE
# # ---------------------------
# st.subheader("📈 Model Performance")

# y_true = filtered_df["IsFraud"]
# y_scores = filtered_df["FraudProbability"]
# y_pred = (y_scores > 0.5).astype(int)

# col4, col5 = st.columns(2)

# with col4:
#     cm = confusion_matrix(y_true, y_pred)
#     fig_cm = px.imshow(
#         cm,
#         text_auto=True,
#         color_continuous_scale="Blues",
#         labels=dict(x="Predicted", y="Actual", color="Count")
#     )
#     st.plotly_chart(fig_cm, use_container_width=True)

# with col5:
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)

#     fig_roc = go.Figure()
#     fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
#     fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
#     fig_roc.update_layout(
#         title=f"ROC Curve (AUC = {round(roc_auc,2)})",
#         xaxis_title="False Positive Rate",
#         yaxis_title="True Positive Rate"
#     )

#     st.plotly_chart(fig_roc, use_container_width=True)

# st.markdown("---")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import os

# st.set_page_config(layout="wide")

# st.title("🧠 AI Behavior-Based Fraud Detection System")

# # -------------------------------------
# # LOAD DATA
# # -------------------------------------
# @st.cache_data
# def load_data():
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(BASE_DIR, "data", "credit_card_fraud.csv")
#     df = pd.read_csv(file_path)
#     return df

# df = load_data()

# df.rename(columns={"Class": "IsFraud"}, inplace=True)

# # -------------------------------------
# # BEHAVIOR ANALYSIS LOGIC
# # -------------------------------------

# def calculate_risk_score(amount, time_gap, pca_score):

#     risk = 0

#     # High amount risk
#     if amount > 2000:
#         risk += 30
#     elif amount > 1000:
#         risk += 20
#     elif amount > 500:
#         risk += 10

#     # Time anomaly risk
#     if time_gap < 10:
#         risk += 25

#     # PCA abnormality
#     if abs(pca_score) > 5:
#         risk += 30
#     elif abs(pca_score) > 3:
#         risk += 15

#     return min(risk, 100)

# # -------------------------------------
# # USER INPUT PANEL
# # -------------------------------------

# st.sidebar.header("💳 Simulate Transaction")

# amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=500.0)

# time_gap = st.sidebar.slider("Time Gap Since Last Transaction (seconds)", 0, 5000, 60)

# pca_score = st.sidebar.slider("Behavior Deviation Score (PCA based)", -10.0, 10.0, 1.0)

# # -------------------------------------
# # CALCULATE RISK
# # -------------------------------------

# risk_score = calculate_risk_score(amount, time_gap, pca_score)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.metric("💰 Transaction Amount", amount)

# with col2:
#     st.metric("⚡ Time Gap", time_gap)

# with col3:
#     st.metric("🎯 Risk Score (0-100)", risk_score)

# st.markdown("---")

# # -------------------------------------
# # FRAUD DECISION
# # -------------------------------------

# threshold = 60

# if risk_score >= threshold:
#     st.error(f"🚨 HIGH RISK ALERT! Risk Score: {risk_score}/100")
#     fraud_detected = True
# else:
#     st.success(f"✅ Transaction Safe. Risk Score: {risk_score}/100")
#     fraud_detected = False

# # -------------------------------------
# # RISK GAUGE VISUAL
# # -------------------------------------

# fig = px.bar(
#     x=["Risk Score"],
#     y=[risk_score],
#     range_y=[0,100],
#     text=[risk_score],
# )

# fig.update_layout(
#     title="Risk Score Visualization",
#     yaxis_title="Risk Level (0-100)",
# )

# st.plotly_chart(fig, use_container_width=True)

# # -------------------------------------
# # BEHAVIOR INSIGHTS
# # -------------------------------------

# st.subheader("🔍 Behavior Analysis Report")

# reasons = []

# if amount > 1000:
#     reasons.append("High Transaction Amount")

# if time_gap < 10:
#     reasons.append("Very Frequent Transaction")

# if abs(pca_score) > 3:
#     reasons.append("Unusual Behavioral Pattern")

# if len(reasons) == 0:
#     reasons.append("Normal Behavior Pattern")

# for r in reasons:
#     st.write("•", r)

# # -------------------------------------
# # HISTORICAL FRAUD TREND
# # -------------------------------------

# st.markdown("---")
# st.subheader("📊 Historical Fraud Distribution")

# fraud_counts = df["IsFraud"].value_counts().reset_index()
# fraud_counts.columns = ["Label", "Count"]
# fraud_counts["Label"] = fraud_counts["Label"].replace({0: "Legit", 1: "Fraud"})

# fig2 = px.pie(
#     fraud_counts,
#     names="Label",
#     values="Count",
#     hole=0.4,
#     color="Label",
#     color_discrete_map={"Fraud": "red", "Legit": "green"}
# )

# st.plotly_chart(fig2, use_container_width=True)
# # 
# st.markdown("---")

# st.success("🧠 Real-Time Behavior Monitoring Active")
# st.success("✅ Enhanced Credit Card Fraud Dashboard Ready!")