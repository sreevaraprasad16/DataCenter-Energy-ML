import json
import os
import random
import time

import paho.mqtt.client as mqtt


BROKER = os.getenv(
    "MQTT_BROKER",
    "test.mosquitto.org"
)

PORT = int(
    os.getenv(
        "MQTT_PORT",
        "1883"
    )
)

TOPIC = os.getenv(
    "MQTT_TOPIC",
    "datacenter/energy/telemetry"
)

DEVICE_ID = os.getenv(
    "DEVICE_ID",
    "DC-SENSOR-01"
)


def generate_sensor_data():
    temperature = round(
        random.uniform(20, 30),
        2
    )

    humidity = round(
        random.uniform(40, 60),
        2
    )

    it_load = round(
        random.uniform(100, 250),
        2
    )

    cooling_power = round(
        it_load * random.uniform(0.15, 0.30),
        2
    )

    total_power = round(
        it_load
        + cooling_power
        + random.uniform(5, 15),
        2
    )

    pue = round(
        total_power / it_load,
        3
    )

    return {
        "Device_ID": DEVICE_ID,
        "Timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Temperature_C": temperature,
        "Humidity_Percent": humidity,
        "IT_Load_kW": it_load,
        "Cooling_Power_kW": cooling_power,
        "Total_Power_kW": total_power,
        "PUE": pue
    }


def on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties
):
    if reason_code == 0:
        print("Connected to MQTT broker!")
        print("Publishing to:", TOPIC)
    else:
        print(
            "MQTT connection failed:",
            reason_code
        )


def on_disconnect(
    client,
    userdata,
    disconnect_flags,
    reason_code,
    properties
):
    print(
        "Disconnected from MQTT broker.",
        "Reason:",
        reason_code
    )


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=(
        "datacenter-energy-simulator-"
        + DEVICE_ID
    )
)

client.on_connect = on_connect
client.on_disconnect = on_disconnect


print("Starting Data Center Sensor Simulator...")
print("MQTT Broker:", BROKER)
print("MQTT Port:", PORT)
print("MQTT Topic:", TOPIC)
print("Device ID:", DEVICE_ID)
print("Publishing interval: 10 seconds")
print("Press Ctrl+C to stop.\n")


try:
    client.connect(
        BROKER,
        PORT,
        60
    )

    client.loop_start()

    while True:
        sensor_data = generate_sensor_data()

        payload = json.dumps(
            sensor_data
        )

        result = client.publish(
            TOPIC,
            payload
        )

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(
                "Published:",
                payload
            )
        else:
            print(
                "Publish failed. MQTT code:",
                result.rc
            )

        time.sleep(10)

except KeyboardInterrupt:
    print("\nSensor simulator stopped.")

finally:
    client.loop_stop()
    client.disconnect()