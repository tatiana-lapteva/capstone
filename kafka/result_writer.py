# kafka/result_writer.py
#
# Service 5: Result Writer
# Reads: Kafka topic predictions.fraud
# Creates predictions table if not exists
# Writes: PostgreSQL predictions table

import json
import traceback
import logging
from datetime import datetime, timezone
from confluent_kafka import Consumer, Producer, KafkaError
from sqlalchemy import create_engine, text
from config import KAFKA_BOOTSTRAP, DB_URL

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [RW] %(levelname)s %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOPIC_IN        = "predictions.fraud"
TOPIC_ERRORS    = "transactions.errors"
GROUP_ID        = "result-writer-group"


consumer = Consumer({
    "bootstrap.servers":  KAFKA_BOOTSTRAP,
    "group.id":           GROUP_ID,
    "auto.offset.reset":  "earliest",
    "enable.auto.commit": False,
})
producer = Producer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,})
engine = create_engine(DB_URL, echo=False)


_metrics = {
    "written":    0,
    "fraud":      0,
    "non_fraud":  0,
    "errors":     0,
}

# DDL
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                        SERIAL PRIMARY KEY,

    -- Transaction reference
    transaction_id            BIGINT NOT NULL,

    -- Core prediction
    fraud_score               DOUBLE PRECISION NOT NULL,
    is_fraud_pred             BOOLEAN          NOT NULL,
    threshold                 DOUBLE PRECISION NOT NULL DEFAULT 0.30,

    -- Model metadata
    model_version             VARCHAR(100),
    hgnn_embedding_type       VARCHAR(50),
    n_tabular_features        INTEGER,
    n_gnn_embedding_features  INTEGER,
    n_final_features          INTEGER,
    latency_ms                DOUBLE PRECISION,

    -- JSONB payloads
    explanation               JSONB,
    graph_context             JSONB,
    raw_tx                    JSONB,

    -- Timestamps
    predicted_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique constraint for ON CONFLICT
CREATE UNIQUE INDEX IF NOT EXISTS predictions_transaction_id_uidx
    ON predictions (transaction_id);

-- Index for analytics queries
CREATE INDEX IF NOT EXISTS predictions_is_fraud_idx
    ON predictions (is_fraud_pred);

CREATE INDEX IF NOT EXISTS predictions_predicted_at_idx
    ON predictions (predicted_at DESC);
"""


def ensure_table() -> None:
    """Creates predictions table if not exists."""
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
    logger.info("Table predictions ready")


def delivery_report(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")


def send_error(
    result: dict | None,
    stage: str,
    error: Exception | str,
    original_topic: str = TOPIC_IN,
) -> None:
    tx_id = result.get("TransactionID") if isinstance(result, dict) else None

    error_event = {
        "stage": stage,
        "TransactionID": tx_id,
        "error_type": type(error).__name__ if isinstance(error, Exception) else "Error",
        "error_message": str(error),
        "original_topic": original_topic,
        "payload": result,
        "traceback": traceback.format_exc() if isinstance(error, Exception) else None,
    }

    producer.produce(
        topic=TOPIC_ERRORS,
        key=str(tx_id) if tx_id is not None else None,
        value=json.dumps(error_event, default=str),
        callback=delivery_report,
    )
    producer.flush()



def save_prediction(result: dict) -> None:
    """
    Writes fraud prediction result to PostgreSQL.
    """
    explanation = result.get("explanation", {})
    graph_context = result.get("graph_context", {})
    raw_tx = result.get("raw_tx", {})

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO predictions (
                    transaction_id,
                    fraud_score,
                    is_fraud_pred,
                    threshold,
                    model_version,
                    hgnn_embedding_type,
                    n_tabular_features,
                    n_gnn_embedding_features,
                    n_final_features,
                    latency_ms,
                    explanation,
                    graph_context,
                    raw_tx,
                    predicted_at
                )
                VALUES (
                    :transaction_id,
                    :fraud_score,
                    :is_fraud_pred,
                    :threshold,
                    :model_version,
                    :hgnn_embedding_type,
                    :n_tabular_features,
                    :n_gnn_embedding_features,
                    :n_final_features,
                    :latency_ms,
                    CAST(:explanation   AS JSONB),
                    CAST(:graph_context AS JSONB),
                    CAST(:raw_tx        AS JSONB),
                    :predicted_at
                )
                ON CONFLICT (transaction_id)
                DO UPDATE SET
                    fraud_score              = EXCLUDED.fraud_score,
                    is_fraud_pred            = EXCLUDED.is_fraud_pred,
                    threshold                = EXCLUDED.threshold,
                    model_version            = EXCLUDED.model_version,
                    hgnn_embedding_type      = EXCLUDED.hgnn_embedding_type,
                    n_tabular_features       = EXCLUDED.n_tabular_features,
                    n_gnn_embedding_features = EXCLUDED.n_gnn_embedding_features,
                    n_final_features         = EXCLUDED.n_final_features,
                    latency_ms               = EXCLUDED.latency_ms,
                    explanation              = EXCLUDED.explanation,
                    graph_context            = EXCLUDED.graph_context,
                    raw_tx                   = EXCLUDED.raw_tx,
                    predicted_at             = EXCLUDED.predicted_at
            """),
            {
                "transaction_id":           result["TransactionID"],
                "fraud_score":              result["fraud_score"],
                "is_fraud_pred":            result["is_fraud"],
                "threshold":                result.get("threshold", 0.30),
                "model_version":            result.get("model_version"),
                "hgnn_embedding_type":      result.get("hgnn_embedding_type"),
                "n_tabular_features":       result.get("n_tabular_features"),
                "n_gnn_embedding_features": result.get("n_gnn_embedding_features"),
                "n_final_features":         result.get("n_final_features"),
                "latency_ms":               result.get("latency_ms"),
                "explanation":              json.dumps(explanation, default=str),
                "graph_context":            json.dumps(graph_context, default=str),
                "raw_tx":                   json.dumps(raw_tx, default=str),
                "created_at":               datetime.now(timezone.utc),
            })


def main():
    ensure_table()
    consumer.subscribe([TOPIC_IN])

    logger.info(f"Result Writer started | {TOPIC_IN} → PostgreSQL")
    logger.info(f"   Errors: {TOPIC_ERRORS}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Kafka consumer error: {msg.error()}")
                continue

            result = None
            try:
                result = json.loads(msg.value())
                tx_id = result.get("TransactionID")
                save_prediction(result)

                is_fraud = result.get("is_fraud", False)
                _metrics["written"] += 1
                if is_fraud:
                    _metrics["fraud"] += 1
                else:
                    _metrics["non_fraud"] += 1

                logger.info(
                    f"{'🚨 FRAUD' if is_fraud else 'OK    '} | "
                    f"tx={tx_id} "
                    f"score={result['fraud_score']:.4f} "
                    f"latency={result.get('latency_ms', 0):.1f}ms"
                )

                # Every 50 transactions
                if _metrics["written"] % 50 == 0:
                    total = _metrics["written"]
                    logger.info(
                        f"Metrics | written={total} "
                        f"fraud={_metrics['fraud']} "
                        f"fraud_rate={_metrics['fraud']/max(total,1):.3f} "
                        f"errors={_metrics['errors']}"
                    )
                consumer.commit(msg)

            except Exception as e:
                _metrics["errors"] += 1
                logger.error(
                    f"Write failed tx="
                    f"{result.get('TransactionID') if result else '?'}: {e}"
                )
                send_error(
                    result=result,
                    stage="result_writer",
                    error=e,
                    original_topic=TOPIC_IN,
                )
                consumer.commit(msg)

    except KeyboardInterrupt:
        logger.info("Stopped by user")

    finally:
        consumer.close()
        logger.info(
            f"Consumer closed | "
            f"written={_metrics['written']} "
            f"fraud={_metrics['fraud']} "
            f"errors={_metrics['errors']}"
        )


if __name__ == "__main__":
    main()