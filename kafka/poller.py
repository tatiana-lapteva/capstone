# kafka/poller.py

import json
import math
from datetime import datetime, date
from typing import Any
from sqlalchemy import create_engine, text
from confluent_kafka import Producer
from config import KAFKA_BOOTSTRAP, DB_URL

# Config
TOPIC           = "transactions.raw"
BATCH_SIZE      = 3   # TEST

# Kafka Producer
producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

def delivery_report(err, msg):
    if err:
        print(f"[WARN] Delivery failed: {err}")
    else:
        print(f"[DONE] Sent: topic={msg.topic()}  "
              f"partition={msg.partition()}  "
              f"offset={msg.offset()}")

def to_json_safe(value: Any):
    if value is None:
        return None

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    return value

def fetch_transactions(engine, batch_size: int) -> list[dict]:
    """Reads unprocessed transactions from Postgres."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT *
            FROM transactions
            WHERE is_processed = FALSE
            ORDER BY created_at ASC
            LIMIT :batch_size
            FOR UPDATE SKIP LOCKED
        """), {"batch_size": batch_size}).fetchall()

    return [dict(r._mapping) for r in rows]


def mark_as_processed(engine, transaction_ids: list[int]) -> None:
    if not transaction_ids:
        return

    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE transactions
                SET is_processed = TRUE
                WHERE "TransactionID" = ANY(:ids)
            """),
            {"ids": transaction_ids},
        )


def send_to_kafka(transactions: list[dict]) -> list[int]:
    """Send transations to Kafka. """
    sent_ids = []
    for tx in transactions:
        # convert to JSON
        tx_json = {
            k: (v if v is not None and not (isinstance(v, float) and v != v) else None)
            for k, v in tx.items()
        }
        tx_id = tx_json["TransactionID"]
        # datetime → string
        for k, v in tx_json.items():
            if hasattr(v, "isoformat"):
                tx_json[k] = v.isoformat()

        producer.produce(
            topic    = TOPIC,
            key      = str(tx_id),
            value    = json.dumps(tx_json),
            callback = delivery_report,
        )
        sent_ids.append(tx_id)

    producer.flush()
    return sent_ids


def main():
    print(f"[DONE] Poller started")
    print(f"   DB:         {DB_URL.split('@')[1]}")
    print(f"   Kafka:      {KAFKA_BOOTSTRAP}")
    print(f"   Topic:      {TOPIC}")
    print(f"   Batch size: {BATCH_SIZE}\n")

    engine = create_engine(DB_URL, echo=False)

    transactions = fetch_transactions(engine, BATCH_SIZE)

    if not transactions:
        print("[WARN]  No unprocessed transactions found")
        return

    print(f" Fetched {len(transactions)} transactions from Postgres")
    for tx in transactions:
        print(f"   TransactionID={tx['TransactionID']}  "
              f"Amt={tx['TransactionAmt']}  "
              f"DT={tx['TransactionDT']}")

    print(f"\n Sending to Kafka topic: {TOPIC}")
    try:
        sent_ids = send_to_kafka(transactions)
        mark_as_processed(engine, sent_ids)
    except Exception as e:
        print(f"[ERROR] Poller failed before marking processed: {e}")
        return

    print(f"\n [DONE]: {len(transactions)} transactions sent")

if __name__ == "__main__":
    main()