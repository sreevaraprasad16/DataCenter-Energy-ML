import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/data_center_energy.csv")

# Create scatter plot
plt.scatter(df["IT_Load_kW"], df["PUE"])

plt.xlabel("IT Load (kW)")
plt.ylabel("PUE")
plt.title("IT Load vs PUE")

plt.show()