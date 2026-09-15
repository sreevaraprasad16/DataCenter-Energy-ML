import pandas as pd
import streamlit as st
from supabase import create_client
from sklearn.linear_model import LinearRegression


# =====================================================
# SUPABASE CONNECTION
# =====================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# =====================================================
# LOAD HISTORICAL PUE DATA
# =====================================================

response = (
    supabase
    .table("monitoring_data")
    .select("Timestamp, PUE")
    .order("created_at", desc=False)
    .limit(1000)
    .execute()
)

data = pd.DataFrame(response.data)


# =====================================================
# CHECK DATA
# =====================================================

if data.empty:
    print("No monitoring data found.")
    raise SystemExit


data["PUE"] = pd.to_numeric(
    data["PUE"],
    errors="coerce"
)

data = data.dropna(
    subset=["PUE"]
).reset_index(drop=True)


# =====================================================
# PREPARE TIME-BASED FEATURES
# =====================================================

data["Time_Index"] = range(
    len(data)
)

X = data[
    ["Time_Index"]
]

y = data[
    "PUE"
]


# =====================================================
# TRAIN FORECASTING MODEL
# =====================================================

model = LinearRegression()

model.fit(
    X,
    y
)


# =====================================================
# FORECAST FUTURE PUE
# =====================================================

future_steps = 6

last_index = data["Time_Index"].iloc[-1]

future_indexes = range(
    last_index + 1,
    last_index + future_steps + 1
)

future_data = pd.DataFrame(
    {
        "Time_Index": future_indexes
    }
)

future_predictions = model.predict(
    future_data
)


# =====================================================
# DISPLAY FORECAST
# =====================================================

print("\n====================================")
print("PUE FORECAST")
print("====================================")

for i, prediction in enumerate(
    future_predictions,
    start=1
):

    print(
        f"Future Reading {i}: "
        f"Predicted PUE = {prediction:.3f}"
    )

print("====================================")