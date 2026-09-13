import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("data/data_center_energy.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================================================
# CREATE MODELS
# =====================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}


# =====================================================
# TRAIN AND EVALUATE
# =====================================================

results = []


for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )


# =====================================================
# DISPLAY RESULTS
# =====================================================

results_df = pd.DataFrame(results)

print("\n====================================")
print("CLEAN MODEL EVALUATION")
print("====================================")

print(
    results_df.round(4).to_string(index=False)
)