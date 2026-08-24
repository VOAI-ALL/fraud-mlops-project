import json
import time

import pandas as pd
from confluent_kafka import Producer


# --------------------------------------------------
# Kafka configuration
# --------------------------------------------------

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "fraud_transactions"


producer = Producer({
    "bootstrap.servers": KAFKA_SERVER
})


# --------------------------------------------------
# Delivery confirmation
# --------------------------------------------------

def delivery_report(err, msg):

    if err is not None:
        print(f"Delivery failed: {err}")
    else:
        print(
            f"Delivered transaction "
            f"to {msg.topic()} [{msg.partition()}]"
        )


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

data = pd.read_csv("data/creditcard.csv")

# Use 100 transactions as simulated live data
test_data = data.iloc[500:600].copy()


# --------------------------------------------------
# Features expected by the fraud API
# --------------------------------------------------

features = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# --------------------------------------------------
# Send transactions to Kafka
# --------------------------------------------------

for i, (_, row) in enumerate(test_data.iterrows(), start=1):

    transaction = {
        column: float(row[column])
        for column in features
    }

    producer.produce(
        KAFKA_TOPIC,
        value=json.dumps(transaction),
        callback=delivery_report
    )

    producer.poll(0)

    print(f"{i}/100 -> Transaction sent to Kafka")

    time.sleep(0.05)


# --------------------------------------------------
# Wait for all messages to be delivered
# --------------------------------------------------

producer.flush()

print("\nFinished sending 100 transactions to Kafka.")