import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

print("🚀 Training Started...")

# -----------------------------------
# LOAD DATA
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data","creditcard.csv")

df = pd.read_csv(file_path)

# Use only 10000 rows
df = df.sample(n=10000, random_state=42)

print("✅ Data Loaded (10000 rows)")

# -----------------------------------
# FEATURES & TARGET
# -----------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# MODEL
# -----------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("✅ Model Training Completed")

# -----------------------------------
# EVALUATION
# -----------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

print("\n📊 Model Performance:")
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC:", round(auc_score, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------------jj
# SAVE MODEL
# -----------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "rf_model_10k.pkl")
joblib.dump(model, MODEL_PATH)

print("\n💾 Model Saved as rf_model_10k.pkl")
print("🎉 Training Finished Successfully!")