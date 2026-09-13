import pandas as pd
import joblib


# =====================================================
# LOAD CLEAN MODEL
# =====================================================

model = joblib.load(
    "random_forest_clean_model.pkl"
)


# =====================================================
# USER INPUT
# =====================================================

temperature = float(
    input("Enter Temperature (°C): ")
)

humidity = float(
    input("Enter Humidity (%): ")
)

it_load = float(
    input("Enter IT Load (kW): ")
)

cooling_power = float(
    input("Enter Cooling Power (kW): ")
)


# =====================================================
# CREATE INPUT DATA
# Total_Power_kW REMOVED
# =====================================================

input_data = pd.DataFrame(
    [
        {
            "Temperature_C": temperature,
            "Humidity_Percent": humidity,
            "IT_Load_kW": it_load,
            "Cooling_Power_kW": cooling_power
        }
    ]
)


# =====================================================
# PREDICT PUE
# =====================================================

prediction = model.predict(
    input_data
)


# =====================================================
# DISPLAY RESULT
# =====================================================

print("\n====================================")
print("PUE PREDICTION")
print("====================================")

print(
    "Predicted PUE:",
    round(prediction[0], 3)
)

print("\nTotal_Power_kW was NOT used for prediction.")