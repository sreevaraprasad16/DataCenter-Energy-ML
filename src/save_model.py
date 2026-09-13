import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/data_center_energy.csv")

# Input features
X = df[
    [
        "Temperature_C",
        "Humidity_Percent",
        "IT_Load_kW",
        "Total_Power_kW",
        "Cooling_Power_kW"
    ]
]

# Target
y = df["PUE"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "random_forest_model.pkl")

print("Random Forest model saved successfully!")