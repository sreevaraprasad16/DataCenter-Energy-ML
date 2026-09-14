import paho.mqtt.client as mqtt


BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "datacenter/energy/telemetry"


def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected to MQTT broker!")
    print("Subscribing to:", TOPIC)

    client.subscribe(TOPIC)


def on_message(client, userdata, message):
    payload = message.payload.decode("utf-8")

    print("\nReceived telemetry:")
    print(payload)


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="datacenter-subscriber-test"
)

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)

print("Waiting for MQTT messages...")
print("Press Ctrl+C to stop.\n")

client.loop_forever()