import json
import urllib.request

from confluent_kafka import Consumer


# --------------------------------------------------
# Kafka configuration
# --------------------------------------------------

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "fraud_transactions"
GROUP_ID = "fraud-api-consumer"


consumer = Consumer({
    "bootstrap.servers": KAFKA_SERVER,
    "group.id": GROUP_ID,
    "auto.offset.reset": "earliest"
})


# --------------------------------------------------
# Subscribe to Kafka topic
# --------------------------------------------------

consumer.subscribe([KAFKA_TOPIC])

print("Kafka Consumer started...")
print(f"Listening to topic: {KAFKA_TOPIC}")
print("Waiting for transactions...\n")


# --------------------------------------------------
# Consume transactions
# --------------------------------------------------

try:

    while True:

        message = consumer.poll(1.0)

        if message is None:
            continue

        if message.error():
            print(f"Kafka error: {message.error()}")
            continue

        # Convert Kafka message to Python dictionary
        transaction = json.loads(
            message.value().decode("utf-8")
        )

        # --------------------------------------------------
        # Send transaction to FastAPI
        # --------------------------------------------------

        request = urllib.request.Request(
            "http://127.0.0.1:8000/predict",
            data=json.dumps(transaction).encode("utf-8"),
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        try:

            with urllib.request.urlopen(request) as response:

                result = json.loads(
                    response.read().decode("utf-8")
                )

                print(
                    f"Kafka transaction -> "
                    f"{result['prediction']} | "
                    f"probability={result['fraud_probability']}"
                )

        except Exception as error:

            print(
                f"FastAPI request failed: {error}"
            )


except KeyboardInterrupt:

    print("\nConsumer stopped.")


finally:

    consumer.close()