import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
import streamlit as st
from supabase import create_client

# ============================================================
# SUPABASE CONNECTION
# ============================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# ============================================================
# LOAD REAL MONITORING DATA
# ============================================================

response = (
    supabase
    .table("monitoring_data")
    .select(
        "Temperature_C,"
        "Humidity_Percent,"
        "IT_Load_kW,"
        "Cooling_Power_kW,"
        "PUE"
    )
    .limit(1000)
    .execute()
)

data = pd.DataFrame(response.data)

print("\n====================================")
print("ML ANOMALY DETECTION")
print("====================================")

print("Rows loaded from Supabase:", len(data))

# ============================================================
# CHECK DATA
# ============================================================

features = [
    "Temperature_C",
    "Humidity_Percent",
    "IT_Load_kW",
    "Cooling_Power_kW",
    "PUE"
]

if data.empty:
    print("No monitoring data found.")
    raise SystemExit

data = data.dropna(
    subset=features
)

print(
    "Rows available for training:",
    len(data)
)

# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

X = data[features]

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X)

# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = "anomaly_detection_model.pkl"

joblib.dump(
    model,
    MODEL_PATH
)

print("\nIsolation Forest trained successfully.")

print("Features used:")
print(features)

print(
    "\nModel saved as:",
    MODEL_PATH
)

# ============================================================
# TEST ANOMALY DETECTION
# ============================================================

data["Anomaly"] = model.predict(X)

data["Anomaly_Status"] = data["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

print("\nAnomaly summary:")

print(
    data["Anomaly_Status"]
    .value_counts()
)

print("\n====================================")
print("Training completed successfully.")
print("====================================")