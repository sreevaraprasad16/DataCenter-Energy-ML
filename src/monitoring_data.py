import pandas as pd
import numpy as np

np.random.seed(42)

n = 100

timestamps = pd.date_range(
    start="2026-09-13 08:00",
    periods=n,
    freq="min"
)

temperature = np.random.uniform(20, 28, n)
humidity = np.random.uniform(40, 60, n)
it_load = np.random.uniform(100, 220, n)

cooling_power = (
    15
    + (temperature - 18) * 1.5
    + it_load * 0.08
    + np.random.normal(0, 2, n)
)

total_power = (
    it_load
    + cooling_power
    + np.random.uniform(5, 15, n)
)

pue = total_power / it_load

df = pd.DataFrame({
    "Timestamp": timestamps,
    "Temperature_C": temperature.round(2),
    "Humidity_Percent": humidity.round(2),
    "IT_Load_kW": it_load.round(2),
    "Cooling_Power_kW": cooling_power.round(2),
    "Total_Power_kW": total_power.round(2),
    "PUE": pue.round(3)
})

df.to_csv(
    "data/monitoring_data.csv",
    index=False
)

print("Monitoring dataset created successfully!")
print("Shape:", df.shape)
print(df.head())