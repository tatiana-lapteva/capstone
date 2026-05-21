# kafka/inference_consumer.py
#
# Service 4: Fraud Inference Consumer
#
# Reads:
#   Kafka topic: graph.updates
# Performs:
#   1. Builds HGNN semantic embeddings for the incoming transaction
#   2. Concatenates tabular features + HGNN semantic embeddings
#   3. Runs XGBoost inference
#   4. Builds prediction explanation
#
# Sends:
#   Kafka topic: predictions.fraud
#
# Errors:
#   Kafka topic: transactions.errors

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from confluent_kafka import Consumer, Producer, KafkaError

PROJECT_DIR = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_DIR / "ml" / "artifacts"

sys.path.append(str(PROJECT_DIR / "ml" / "src"))

import utils
import logging
from feature_store import FeatureStore
from graph_state_manager import GraphStateManager, EMB_DIM, ENTITY_FEAT_COLS_DICT
from incremental_inference import incremental_hgnn_embedding
from explainability import FraudExplainer
from config import KAFKA_BOOTSTRAP

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [INF] %(levelname)s %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

_metrics = {
    "processed":  0,
    "fraud":      0,
    "non_fraud":  0,
    "errors":     0,
    "latency_ms": [],
}


TOPIC_IN        = "graph.updates"
TOPIC_OUT       = "predictions.fraud"
TOPIC_ERRORS    = "transactions.errors"
GROUP_ID        = "inference-group"
THRESHOLD       = 0.30

MODEL_VERSION = "xgb_hgnn_semantic_v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

consumer = Consumer({
    "bootstrap.servers":  KAFKA_BOOTSTRAP,
    "group.id":           GROUP_ID,
    "auto.offset.reset":  "earliest",
    "enable.auto.commit": False,
})
producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP,})

def load_artifacts() -> dict[str, Any]:
    """
    Required artifacts:
      - xgb_gnn_model.pkl
      - xgb_gnn_feature_cols.json
      - semantic_emb_cols.json
      - graph_config.pkl
      - hgnn_model.pt

    These must correspond to batch training:
      X_train_semantic = concat(X_train_cleaned, train_emb[semantic_cols])
      xgb_gnn_model trained on X_train_semantic
    """
    logger.info("Loading artifacts...")
    xgb_model = utils.load_artifact(
        str(ARTIFACT_DIR / "models" / "xgb_gnn_model.pkl")
    )

    xgb_feature_cols = utils.load_artifact(
        str(ARTIFACT_DIR / "xgb_gnn_feature_cols.json")
    )

    semantic_emb_cols = utils.load_artifact(
        str(ARTIFACT_DIR / "semantic_emb_cols.json")
    )

    graph_config = utils.load_artifact(
        str(ARTIFACT_DIR / "hgnn_v3_baseline" / "configs" / "graph_config.pkl")
    )

    hgnn_model_path = ARTIFACT_DIR / "hgnn_v3_baseline" / "models" / "gnn.pt"

    if not hgnn_model_path.exists():
        raise FileNotFoundError(
            f"HGNN model artifact not found: {hgnn_model_path}"
        )

    # Ініціалізуємо модель і завантажуємо ваги
    from graph import HGNN
    from configs import ModelConfig   # якщо є окремий файл конфігів

    model_config = utils.load_artifact(
        str(ARTIFACT_DIR / "hgnn_v3_baseline" / "configs" / "model_config.pkl")
    )


    hgnn_model = HGNN(
        entity_cols   = graph_config.entity_cols,
        in_dim        = len(graph_config.graph_features),
        config        = model_config,
        edge_types    = None,   # буде визначено при першому forward
        edge_attr_dim = graph_config.edge_attr_dim,
    )
    state_dict = utils.load_artifact(str(hgnn_model_path), device=DEVICE)
    hgnn_model.load_state_dict(state_dict)
    hgnn_model.to(DEVICE)
    hgnn_model.eval()

    logger.info(f"  XGBoost features: {len(xgb_feature_cols)}")
    logger.info(f"  Semantic emb cols: {len(semantic_emb_cols)}")
    logger.info(f"  HGNN loaded: {sum(p.numel() for p in hgnn_model.parameters()):,} params")
    logger.info(" Artifacts loaded")

    return {
        "xgb_model":         xgb_model,
        "xgb_feature_cols":  xgb_feature_cols,
        "semantic_emb_cols": semantic_emb_cols,
        "graph_config":      graph_config,
        "hgnn_model":        hgnn_model,
    }


ARTIFACTS         = load_artifacts()
XGB_MODEL         = ARTIFACTS["xgb_model"]
XGB_FEATURE_COLS  = ARTIFACTS["xgb_feature_cols"]
SEMANTIC_EMB_COLS = ARTIFACTS["semantic_emb_cols"]
GRAPH_CONFIG      = ARTIFACTS["graph_config"]
HGNN_MODEL        = ARTIFACTS["hgnn_model"]

# Graph state
feature_store = FeatureStore()
graph_state   = GraphStateManager(
    feature_store         = feature_store,
    emb_dim               = EMB_DIM,
    entity_feat_cols_dict = ENTITY_FEAT_COLS_DICT,
)

# Explainability
explainer = FraudExplainer(
    model        = XGB_MODEL,
    feature_cols = XGB_FEATURE_COLS,
    emb_cols     = SEMANTIC_EMB_COLS,
    top_k        = 10,
)

# Kafka helpers
def delivery_report(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")


def send_error(
    payload: dict | None,
    stage: str,
    error: Exception | str,
    original_topic: str = TOPIC_IN,
) -> None:
    tx_id = payload.get("TransactionID") if isinstance(payload, dict) else None

    error_event = {
        "stage": stage,
        "TransactionID": tx_id,
        "error_type": type(error).__name__ if isinstance(error, Exception) else "Error",
        "error_message": str(error),
        "original_topic": original_topic,
        "payload": payload,
        "traceback": traceback.format_exc() if isinstance(error, Exception) else None,
    }

    producer.produce(
        topic=TOPIC_ERRORS,
        key=str(tx_id) if tx_id is not None else None,
        value=json.dumps(error_event, default=str),
        callback=delivery_report,
    )

    producer.flush()


# Feature construction
def embedding_to_dict(
    embedding: np.ndarray,
    emb_cols: list[str],
) -> dict[str, float]:
    """
    Converts HGNN semantic embedding vector into named columns.

    Must match batch:
      semantic_cols, _ = graph.get_embedding_col_names(model)
    """
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if len(embedding) != len(emb_cols):
        raise ValueError(
            f"Embedding size mismatch: {len(embedding)} != {len(emb_cols)}"
        )

    if np.isnan(embedding).any():
        raise ValueError("NaN detected in HGNN embedding")

    if np.isinf(embedding).any():
        raise ValueError("Inf detected in HGNN embedding")

    emb_dict = dict(zip(emb_cols, embedding.tolist()))
    return emb_dict


# Feature construction
def build_xgb_gnn_input(
    tabular_features: dict[str, Any],
    gnn_embedding: np.ndarray,
) -> pd.DataFrame:
    """
    Builds one-row DataFrame:
      tabular features + HGNN semantic embedding
    Then aligns columns to XGB_FEATURE_COLS from batch training.
    """

    emb_dict = embedding_to_dict(
        embedding=gnn_embedding,
        emb_cols=SEMANTIC_EMB_COLS,
    )
    row = {
        **tabular_features,
        **emb_dict,
    }
    X = pd.DataFrame([row])
    # Align columns
    for col in XGB_FEATURE_COLS:
        if col not in X.columns:
            X[col] = np.nan

    X = X[XGB_FEATURE_COLS]

    # Category dtype
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")

    return X


# Inference
def predict(payload: dict) -> dict:
    start            = time.perf_counter()
    tx_id            = payload.get("TransactionID")
    tabular_features = payload.get("tabular_features", {})
    graph_context    = payload.get("graph_context", {})
    subgraph         = payload.get("subgraph", {})
    raw_tx           = payload.get("raw_tx", {})

    # HGNN embedding
    tx_for_hgnn = {**raw_tx, **tabular_features,}

    try:
        gnn_embedding = incremental_hgnn_embedding(
            tx            = tx_for_hgnn,
            subgraph      = subgraph,
            graph_context = graph_context,
            model         = HGNN_MODEL,
            graph_state   = graph_state,
            graph_config  = GRAPH_CONFIG,
            device        = DEVICE,
        )
        gnn_score = float(np.mean(gnn_embedding))  # scalar для логування
    except Exception as e:
        logger.warning(f"HGNN failed tx={tx_id}: {e} — using zeros")
        gnn_embedding = np.zeros(len(SEMANTIC_EMB_COLS), dtype=np.float32)
        gnn_score     = 0.0

        # XGBoost Inference
        X_final = build_xgb_gnn_input(
            tabular_features=tabular_features,
            gnn_embedding=gnn_embedding,
        )
        fraud_score = float(XGB_MODEL.predict_proba(X_final)[:, 1][0])
        is_fraud = bool(fraud_score > THRESHOLD)

        # Explainability
        try:
            explanation = explainer.explain(
                X = X_final,
                fraud_score = fraud_score,
                threshold = THRESHOLD,
            )
        except Exception as e:
            logger.warning(f"Explainer failed: {e}")
            explanation = {}

        latency_ms = (time.perf_counter() - start) * 1000

        return {
        "TransactionID":            tx_id,
        "fraud_score":              round(fraud_score, 6),
        "is_fraud":                 is_fraud,
        "threshold":                THRESHOLD,
        "gnn_score":                round(gnn_score, 6),
        "model_version":            MODEL_VERSION,
        "hgnn_embedding_type":      "semantic",
        "n_tabular_features":       len(tabular_features),
        "n_gnn_embedding_features": len(SEMANTIC_EMB_COLS),
        "n_final_features":         len(XGB_FEATURE_COLS),
        "latency_ms":               round(latency_ms, 3),
        "explanation":              explanation,
        "graph_context":            graph_context,
        "raw_tx":                   raw_tx,
    }

    _metrics["processed"] += 1
    if is_fraud:
        _metrics["fraud"] += 1
    else:
        _metrics["non_fraud"] += 1
    _metrics["latency_ms"].append(latency_ms)

    # Every 10 transactions:
    if _metrics["processed"] % 10 == 0:
        lat = _metrics["latency_ms"]
        logger.info(
            f"Metrics | processed={_metrics['processed']} "
            f"fraud={_metrics['fraud']} "
            f"fraud_rate={_metrics['fraud']/max(_metrics['processed'],1):.3f} "
            f"avg_latency={sum(lat)/len(lat):.1f}ms "
            f"errors={_metrics['errors']}"
        )

    return {
        "TransactionID": tx_id,

        "fraud_score": round(fraud_score, 6),
        "is_fraud": is_fraud,
        "threshold": THRESHOLD,

        "model_version": MODEL_VERSION,
        "latency_ms": round(latency_ms, 3),

        "xgb_model": "xgb_gnn_model",
        "hgnn_embedding_type": "semantic",
        "n_tabular_features": len(tabular_features),
        "n_gnn_embedding_features": len(SEMANTIC_EMB_COLS),
        "n_final_features": len(XGB_FEATURE_COLS),

        "explanation": explanation,

        "graph_context": graph_context,
        "raw_tx": raw_tx,
    }


def main():
    consumer.subscribe([TOPIC_IN])
    logger.info(f"Inference Consumer started | {TOPIC_IN} → {TOPIC_OUT}")
    logger.info(f"  Model:    {MODEL_VERSION}")
    logger.info(f"  XGB cols: {len(XGB_FEATURE_COLS)}")
    logger.info(f"  GNN cols: {len(SEMANTIC_EMB_COLS)}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Kafka error: {msg.error()}")
                continue
            payload = None
            try:
                payload = json.loads(msg.value())
                tx_id   = payload.get("TransactionID")
                result = predict(payload)
                is_fraud = result["is_fraud"]
                _metrics["processed"] += 1
                _metrics["latency_ms"].append(result["latency_ms"])
                if is_fraud:
                    _metrics["fraud"] += 1

                logger.info(
                    f"{'🚨 FRAUD' if is_fraud else 'OK    '} | "
                    f"tx={tx_id} "
                    f"score={result['fraud_score']:.4f} "
                    f"latency={result['latency_ms']:.1f}ms"
                )

                # Every 10 transactions
                if _metrics["processed"] % 10 == 0:
                    lat = _metrics["latency_ms"]
                    logger.info(
                        f"Metrics | "
                        f"processed={_metrics['processed']} "
                        f"fraud={_metrics['fraud']} "
                        f"fraud_rate={_metrics['fraud']/max(_metrics['processed'],1):.3f} "
                        f"avg_latency={sum(lat)/len(lat):.1f}ms "
                        f"errors={_metrics['errors']}"
                    )

                producer.produce(
                    topic    = TOPIC_OUT,
                    key      = str(tx_id),
                    value    = json.dumps(result, default=str),
                    callback = delivery_report,
                )
                producer.flush()
                consumer.commit(msg)
            except Exception as e:
                _metrics["errors"] += 1
                logger.error(
                    f"Inference failed "
                    f"tx={payload.get('TransactionID') if payload else '?'}: {e}"
                )
                send_error(payload, "inference_consumer", e)
                consumer.commit(msg)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        consumer.close()
        lat = _metrics["latency_ms"]
        logger.info(
            f"Consumer closed | "
            f"processed={_metrics['processed']} "
            f"fraud={_metrics['fraud']} "
            f"avg_latency={sum(lat)/len(lat) if lat else 0:.1f}ms "
            f"errors={_metrics['errors']}"
        )


if __name__ == "__main__":
    main()