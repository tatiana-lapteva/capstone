
# check_topics.py
from confluent_kafka.admin import AdminClient
from config import KAFKA_BOOTSTRAP


EXPECTED_TOPICS = [
    "transactions.raw",
    "transactions.features",
    "graph.updates",
    "predictions.fraud",
    "transactions.errors",
]

def check_topics():
    admin  = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
    topics = admin.list_topics(timeout=5).topics

    print(f"Kafka topics ({len(topics)} total):")
    for name in EXPECTED_TOPICS:
        status = "[DONE]" if name in topics else "[WARN] MISSING"
        print(f"  {status} {name}")

if __name__ == "__main__":
    check_topics()