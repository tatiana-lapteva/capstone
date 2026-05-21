# fe_consumer.py
      

import json
from confluent_kafka import Consumer, Producer, KafkaError

from pathlib import Path
import sys
import traceback
import logging

PROJECT_DIR = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_DIR / "ml" / "artifacts"

sys.path.append(str(PROJECT_DIR / "ml" / "src"))
import utils
from feature_store import FeatureStore
from realtime_features import RealTimeFeatureEngine
from config import KAFKA_BOOTSTRAP

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [FE] %(levelname)s %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


TOPIC_IN         = "transactions.raw"
TOPIC_OUT        = "transactions.features"
TOPIC_ERRORS     = "transactions.errors"
GROUP_ID         = "fe-consumer-group"


pipeline_artifacts = utils.load_artifact(
    str(ARTIFACT_DIR / "pipeline_artifacts.pkl"))
drop_cols = utils.load_artifact(
    str(ARTIFACT_DIR / "zero_importance_features.json")
)

feature_engine = RealTimeFeatureEngine(pipeline_artifacts, drop_cols)
feature_store = FeatureStore()

consumer = Consumer({
    "bootstrap.servers":  KAFKA_BOOTSTRAP,
    "group.id":           GROUP_ID,
    "auto.offset.reset":  "earliest",
    "enable.auto.commit": False,   
})

producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
# feature_store = FeatureStore()

_metrics = {"processed": 0, "errors": 0, "fe_errors": 0}

def delivery_report(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")
    else:
        print(f" [DONE] Forwarded to {msg.topic()}  offset={msg.offset()}")


def safe_value(tx: dict, col: str) -> str:
    value = tx.get(col, "unknown")
    if value is None or str(value) == "nan":
        return "unknown"
    return str(value)


def get_user_id(tx: dict) -> str:
    return f"{safe_value(tx, 'card1')}_{safe_value(tx, 'DeviceInfo')}"


def get_card_id(tx: dict) -> str:
    return (
        f"{safe_value(tx, 'card1')}_"
        f"{safe_value(tx, 'card2')}_"
        f"{safe_value(tx, 'card3')}_"
        f"{safe_value(tx, 'card5')}")

def get_device_id(tx: dict) -> str:
    return f"{safe_value(tx, 'DeviceInfo')}_{safe_value(tx, 'DeviceType')}"


def get_addr_id(tx: dict) -> str:
    return f"{safe_value(tx, 'addr1')}_{safe_value(tx, 'addr2')}"

def send_error(
    tx: dict | None,
    stage: str,
    error: Exception | str,
    original_topic: str = TOPIC_IN,
) -> None:
    """
    Send failed transaction/event to transactions.errors.
    """
    tx_id = tx.get("TransactionID") if isinstance(tx, dict) else None

    error_event = {
        "stage": stage,
        "TransactionID": tx_id,
        "error_type": type(error).__name__ if isinstance(error, Exception) else "Error",
        "error_message": str(error),
        "original_topic": original_topic,
        "raw_tx": tx,
        "traceback": traceback.format_exc() if isinstance(error, Exception) else None,
    }

    producer.produce(
        topic=TOPIC_ERRORS,
        key=str(tx_id) if tx_id is not None else None,
        value=json.dumps(error_event, default=str),
        callback=delivery_report,
    )
    producer.flush()


def main():
    consumer.subscribe([TOPIC_IN])
    logger.info(f" FE Consumer started | {TOPIC_IN} -> {TOPIC_OUT}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Kafka error: {msg.error()}")
                continue
            tx = None
            try:
                # Get transaction
                tx = json.loads(msg.value())
                tx_id = tx.get("TransactionID")
                # print(f"Received: TransactionID={tx_id}"
                #     f"Amt={tx.get('TransactionAmt')}"
                #     f"DT={tx.get('TransactionDT')}")

                # Redis context
                user_id = get_user_id(tx)
                card_id = get_card_id(tx)
                device_id = get_device_id(tx)
                addr_id   = get_addr_id(tx)

                redis_context = feature_store.get_context(
                    user_id=user_id,
                    card_id=card_id,)

                # Feature Engineering 
                X = feature_engine.transform(tx, redis_context)
                features_dict = X.iloc[0].to_dict()
            
                out = {
                        "TransactionID": tx_id,
                        "tabular_features": features_dict,
                        "graph_context": {
                            "user_id": user_id,
                            "card_id": card_id,
                            "device_id": device_id,
                            "addr_id": addr_id,
                            "recent_user_txs": redis_context["recent_user_txs"],
                            "recent_card_txs": redis_context["recent_card_txs"],
                        },
                        "raw_tx": {
                            "TransactionID": tx_id,
                            "TransactionDT": tx.get("TransactionDT"),
                            "TransactionAmt": tx.get("TransactionAmt"),
                            "card1": tx.get("card1"),
                            "card2": tx.get("card2"),
                            "card3": tx.get("card3"),
                            "card5": tx.get("card5"),
                            "DeviceInfo": tx.get("DeviceInfo"),
                            "DeviceType": tx.get("DeviceType"),
                            "addr1": tx.get("addr1"),
                            "addr2": tx.get("addr2"),
                        },
                }
                producer.produce(
                    topic=TOPIC_OUT,
                    key=str(tx_id),
                    value=json.dumps(out, default=str),
                    callback=delivery_report,
                )
                producer.flush()

                # Important: update Redis AFTER feature generation.
                # Current transaction must not leak into its own features.
                feature_store.update_after_prediction(
                    user_id=user_id,
                    card_id=card_id,
                    tx=tx,
                )
                _metrics["processed"] += 1
                if _metrics["processed"] % 100 == 0:
                    logger.info(
                        f"Metrics | processed={_metrics['processed']} "
                        f"errors={_metrics['errors']} "
                        f"fe_errors={_metrics['fe_errors']}"
                    )

                logger.info(
                    f"OK | tx={tx_id} "
                    f"amt={tx.get('TransactionAmt')} "
                    f"features={len(features_dict)}"
                )
                consumer.commit(msg)

            except Exception as e:
                _metrics["errors"] += 1
                logger.error(f"FE failed tx={tx.get('TransactionID') if tx else '?'}: {e}")
                send_error(tx, stage="feature_engineering", error=e)
                consumer.commit(msg)

    except KeyboardInterrupt:
        logger.info("Stopped by user")

    finally:
        consumer.close()
        logger.info(
            f"Consumer closed | "
            f"processed={_metrics['processed']} "
            f"errors={_metrics['errors']}"
        )

if __name__ == "__main__":
    main()