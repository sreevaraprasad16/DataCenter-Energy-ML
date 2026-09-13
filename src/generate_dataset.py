import pandas as pd
import numpy as np

# Make the results reproducible
np.random.seed(42)

# Number of records
n = 1000

# Generate timestamps
timestamps = pd.date_range(
    start="2026-09-01 00:00",
    periods=n,
    freq="h"
)

# Generate data-center conditions
temperature = np.random.uniform(18, 30, n)
humidity = np.random.uniform(35, 65, n)
it_load = np.random.uniform(80, 250, n)

# Cooling power increases with temperature and IT load
cooling_power = (
    15
    + (temperature - 18) * 1.5
    + it_load * 0.08
    + np.random.normal(0, 2, n)
)

# Total power = IT power + cooling + other facility power
total_power = (
    it_load
    + cooling_power
    + np.random.uniform(5, 15, n)
)

# PUE = Total Facility Power / IT Equipment Power
pue = total_power / it_load

# Create DataFrame
df = pd.DataFrame({
    "Timestamp": timestamps,
    "Temperature_C": temperature.round(2),
    "Humidity_Percent": humidity.round(2),
    "IT_Load_kW": it_load.round(2),
    "Total_Power_kW": total_power.round(2),
    "Cooling_Power_kW": cooling_power.round(2),
    "PUE": pue.round(3)
})

# Save dataset
df.to_csv("data/data_center_energy.csv", index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print(df.head())