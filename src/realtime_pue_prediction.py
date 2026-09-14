import json
import joblib
import pandas as pd
import paho.mqtt.client as mqtt
import streamlit as st

from supabase import create_client


# =====================================================
# CONFIGURATION
# =====================================================

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "datacenter/energy/telemetry"

MODEL_PATH = "random_forest_clean_model.pkl"


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# =====================================================
# LOAD RANDOM FOREST MODEL
# =====================================================

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")
print("Expected features:")
print(model.feature_names_in_)


# =====================================================
# MQTT CONNECT
# =====================================================

def on_connect(client, userdata, flags, reason_code, properties):

    print("\nConnected to MQTT broker!")
    print("Subscribing to:", TOPIC)

    client.subscribe(TOPIC)


# =====================================================
# RECEIVE TELEMETRY
# =====================================================

def on_message(client, userdata, message):

    try:

        payload = message.payload.decode("utf-8")

        data = json.loads(payload)

        # -------------------------------------------------
        # Extract ML features
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [
                {
                    "Temperature_C": data["Temperature_C"],
                    "Humidity_Percent": data["Humidity_Percent"],
                    "IT_Load_kW": data["IT_Load_kW"],
                    "Cooling_Power_kW": data["Cooling_Power_kW"]
                }
            ]
        )

        # -------------------------------------------------
        # Predict PUE
        # -------------------------------------------------

        prediction = model.predict(input_data)

        predicted_pue = round(float(prediction[0]), 3)

        # -------------------------------------------------
        # Find matching Supabase row
        # -------------------------------------------------

        result = (
            supabase.table("monitoring_data")
            .select("id")
            .eq("Timestamp", data["Timestamp"])
            .limit(1)
            .execute()
        )

        # -------------------------------------------------
        # Save predicted PUE
        # -------------------------------------------------

        if result.data:

            row_id = result.data[0]["id"]

            supabase.table("monitoring_data").update(
                {
                    "Predicted_PUE": predicted_pue
                }
            ).eq("id", row_id).execute()

            print("Predicted PUE saved to Supabase.")

        else:

            print("Matching Supabase row not found.")

        # -------------------------------------------------
        # Display result
        # -------------------------------------------------

        print("\n====================================")
        print("REAL-TIME PUE PREDICTION")
        print("====================================")

        print("Timestamp:", data["Timestamp"])
        print("Temperature:", data["Temperature_C"], "°C")
        print("Humidity:", data["Humidity_Percent"], "%")
        print("IT Load:", data["IT_Load_kW"], "kW")
        print("Cooling Power:", data["Cooling_Power_kW"], "kW")

        print("------------------------------------")

        print("Actual PUE:", data["PUE"])
        print("Predicted PUE:", predicted_pue)

        print("====================================")


    except Exception as error:

        print("\nPrediction error:")
        print(error)


# =====================================================
# MQTT CLIENT
# =====================================================

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="datacenter-realtime-pue-ml"
)

client.on_connect = on_connect
client.on_message = on_message


# =====================================================
# START SERVICE
# =====================================================

print("\nStarting Real-Time PUE Prediction Service...")
print("Waiting for telemetry...")
print("Press Ctrl+C to stop.\n")


client.connect(
    BROKER,
    PORT,
    60
)

client.loop_forever()