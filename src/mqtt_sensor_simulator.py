import json
import random
import time

import paho.mqtt.client as mqtt


BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "datacenter/energy/telemetry"


def generate_sensor_data():
    temperature = round(random.uniform(20, 30), 2)
    humidity = round(random.uniform(40, 60), 2)
    it_load = round(random.uniform(100, 250), 2)
    cooling_power = round(
        it_load * random.uniform(0.15, 0.30),
        2
    )

    total_power = round(
        it_load + cooling_power + random.uniform(5, 15),
        2
    )

    pue = round(
        total_power / it_load,
        3
    )

    return {
    "Device_ID": "DC-SENSOR-01",
    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature_C": temperature,
        "Humidity_Percent": humidity,
        "IT_Load_kW": it_load,
        "Cooling_Power_kW": cooling_power,
        "Total_Power_kW": total_power,
        "PUE": pue
    }


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="datacenter-energy-simulator"
)

client.connect(BROKER, PORT, 60)
client.loop_start()

print("Connected to MQTT broker!")
print("Publishing data to:", TOPIC)
print("Press Ctrl+C to stop.\n")


while True:
    sensor_data = generate_sensor_data()

    payload = json.dumps(sensor_data)

    client.publish(
        TOPIC,
        payload
    )

    print("Published:", payload)

    time.sleep(10)