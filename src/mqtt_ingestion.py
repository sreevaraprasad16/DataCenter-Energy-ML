import json

import paho.mqtt.client as mqtt
from supabase import create_client
import streamlit as st


BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "datacenter/energy/telemetry"


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


REQUIRED_FIELDS = [
    "Timestamp",
    "Temperature_C",
    "Humidity_Percent",
    "IT_Load_kW",
    "Cooling_Power_kW",
    "Total_Power_kW",
    "PUE"
]


def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected to MQTT broker!")
    print("Subscribing to:", TOPIC)

    client.subscribe(TOPIC)


def on_message(client, userdata, message):
    try:
        payload = message.payload.decode("utf-8")

        data = json.loads(payload)

        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in data
        ]

        if missing_fields:
            print(
                "Rejected reading. Missing fields:",
                missing_fields
            )
            return

        supabase.table(
            "monitoring_data"
        ).insert(data).execute()

        print("Stored in Supabase:")
        print(data)

    except Exception as error:
        print("Error processing telemetry:")
        print(error)


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="datacenter-supabase-ingestion"
)

client.on_connect = on_connect
client.on_message = on_message

client.connect(
    BROKER,
    PORT,
    60
)

print("Starting MQTT → Supabase ingestion service...")
print("Waiting for telemetry...")
print("Press Ctrl+C to stop.\n")

client.loop_forever()