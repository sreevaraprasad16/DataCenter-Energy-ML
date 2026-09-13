import pandas as pd
import matplotlib.pyplot as plt
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
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Get feature importance
importance = model.feature_importances_

# Display feature importance
for feature, value in zip(X.columns, importance):
    print(feature, ":", round(value, 4))

# Create graph
plt.bar(X.columns, importance)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()