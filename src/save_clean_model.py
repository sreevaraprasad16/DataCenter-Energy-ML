import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("data/data_center_energy.csv")


# =====================================================
# INPUT FEATURES
# Total_Power_kW REMOVED
# =====================================================

X = df[
    [
        "Temperature_C",
        "Humidity_Percent",
        "IT_Load_kW",
        "Cooling_Power_kW"
    ]
]


# =====================================================
# TARGET
# =====================================================

y = df["PUE"]


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# CREATE RANDOM FOREST
# =====================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(X_train, y_train)


# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "random_forest_clean_model.pkl"
)


print("Clean Random Forest model trained successfully!")
print("Total_Power_kW is NOT used as an input feature.")
print("Model saved as: random_forest_clean_model.pkl")