# credit-fraud-app
🧠 AI-powered Behaviour-Based Credit Card Fraud Detection system using Random Forest (10K dataset) with real-time risk scoring, anomaly analysis, and interactive Streamlit dashboard.
# 🧠 AI Behaviour-Based Credit Card Fraud Detection (10K Dataset)

An advanced machine learning-based fraud detection system built using the Kaggle Credit Card Fraud dataset.  
The system analyzes transaction behaviour patterns and assigns a dynamic fraud risk score (0–100) in real-time using a trained Random Forest model.

---

## 🚀 Live Features

- ✅ Behaviour-based fraud detection
- ✅ Risk scoring system (0–100 scale)
- ✅ Real-time transaction simulation
- ✅ Fraud / Legit classification
- ✅ Interactive Streamlit dashboard
- ✅ Feature importance visualization
- ✅ Behaviour explanation report
- ✅ Cloud deployment ready (Render)

---

## 📊 Dataset

- Source: Kaggle Credit Card Fraud Detection Dataset
- Features: PCA-transformed variables (V1–V28)
- Additional columns: `Time`, `Amount`, `Class`
- Trained on: 10,000 sampled transactions
- Dataset nature: Highly imbalanced (real-world fraud scenario)

---

## 🧠 Model Details

- Algorithm: Random Forest Classifier
- Handles imbalanced fraud data
- Uses behavioural PCA features instead of only transaction amount
- Fraud probability converted into Risk Score (0–100)

---

## 🖥 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Render (Deployment)

---

## ⚙️ How It Works

1. User inputs transaction amount and selects behaviour pattern.
2. System generates behavioural PCA features.
3. Trained model predicts fraud probability.
4. Risk score (0–100) is calculated.
5. Dashboard displays:
   - Risk meter (Gauge)
   - Fraud / Legit classification
   - Behaviour explanation
   - Top feature importance graph

---

## 📈 Feature Importance

The Random Forest model calculates feature importance based on impurity reduction across decision trees.  
Higher importance indicates stronger influence in fraud detection.

---
