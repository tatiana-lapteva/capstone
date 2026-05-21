# create_topics.py

from confluent_kafka.admin import AdminClient, NewTopic
from config import KAFKA_BOOTSTRAP

TOPICS = [
    # Raw transactions from Postgres
    NewTopic("transactions.raw",      num_partitions=1, replication_factor=1),
    # After feature engineering
    NewTopic("transactions.features", num_partitions=1, replication_factor=1),
    # Graph Updates
    NewTopic("graph.updates",         num_partitions=1, replication_factor=1),
    # Inference Results
    NewTopic("predictions.fraud",     num_partitions=1, replication_factor=1),
    # Preprocess Errors
    NewTopic("transactions.errors", num_partitions=1, replication_factor=1)
]

def create_topics():
    admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})

    result = admin.create_topics(TOPICS)

    for topic, future in result.items():
        try:
            future.result()
            print(f"[DONE] Created: {topic}")
        except Exception as e:
            print(f"[WARN] {topic}: {e}")

if __name__ == "__main__":
    print("Creating Kafka topics...")
    create_topics()
    print("Done")