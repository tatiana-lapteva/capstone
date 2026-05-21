# kafka/graph_state_manager.py
#
# Service 3: Graph State Manager
# Reads from Kafka topic: transactions.features
# Update graph state in Redis:
#   - entity node features
#   - adjacency lists
#   - cached embeddings
# Send to: Kafka topic: graph.updates


import json
import numpy as np
import sys
import traceback
from pathlib import Path

from typing import Any
from confluent_kafka import Consumer, Producer, KafkaError

PROJECT_DIR = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_DIR / "ml" / "artifacts"
sys.path.append(str(PROJECT_DIR / "ml" / "src"))
import utils

from feature_store import FeatureStore
from config import KAFKA_BOOTSTRAP


TOPIC_IN        = "transactions.features"
TOPIC_OUT       = "graph.updates"
TOPIC_ERRORS    = "transactions.errors"
GROUP_ID        = "graph-state-group"

# Size of the entity adjacency list
MAX_NEIGHBORS   = 50    # entitie's last transactions
SECONDS_PER_DAY         = 86400 # 1 day — TTL for cached embeddings
EMBEDDING_TTL = 1 * SECONDS_PER_DAY
GRAPH_STATE_TTL = 3 * SECONDS_PER_DAY


def load_graph_runtime_config() -> dict:
    """
    Loads graph runtime metadata saved during batch training.

    Expected optional artifact:
      ml/artifacts/graph_runtime_config.pkl

    Recommended structure:
    {
        "emb_dim": 64,
        "entity_feat_cols_dict": {
            "user_id": [...],
            "device_id": [...],
            "addr_id": [...],
            "card_id": [...]
        }
    }

    Falls back to the current batch GraphConfig settings.
    """
    default_entity_feat_cols_dict = {
        "user_id": [
            "log_total_tx",
            "amt_mean",
            "amt_std",
            "amt_max",
            "amt_min",
            "unique_devices_per_user",
            "time_span",
            "fraud_rate_oof",
        ],
        "device_id": [
            "log_total_tx",
            "amt_mean",
            "amt_std",
            "amt_max",
            "amt_min",
            "unique_users_per_device",
            "time_span",
            "fraud_rate_oof",
        ],
        "addr_id": [
            "log_total_tx",
            "amt_mean",
            "amt_std",
            "amt_max",
            "amt_min",
            "time_span",
            "fraud_rate_oof",
        ],
        "card_id": [
            "log_total_tx",
            "amt_mean",
            "amt_std",
            "amt_max",
            "amt_min",
            "time_span",
            "fraud_rate_oof",
        ],
    }

    config_path = ARTIFACT_DIR / "graph_runtime_config.pkl"

    if config_path.exists():
        config = utils.load_artifact(str(config_path))
        return {
            "emb_dim": int(config.get("emb_dim", 64)),
            "entity_feat_cols_dict": config.get(
                "entity_feat_cols_dict",
                default_entity_feat_cols_dict,
            ),
        }

    return {
        "emb_dim": 64,
        "entity_feat_cols_dict": default_entity_feat_cols_dict,
    }

GRAPH_RUNTIME_CONFIG = load_graph_runtime_config()
EMB_DIM = GRAPH_RUNTIME_CONFIG["emb_dim"]
ENTITY_FEAT_COLS_DICT = GRAPH_RUNTIME_CONFIG["entity_feat_cols_dict"]

consumer = Consumer({
    "bootstrap.servers":  KAFKA_BOOTSTRAP,
    "group.id":           GROUP_ID,
    "auto.offset.reset":  "earliest",
    "enable.auto.commit": False,
})
producer      = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
feature_store = FeatureStore()


def delivery_report(err, msg):
    if err:
        print(f"[ERROR] Delivery: {err}")
    else:
        print(f"[DONE] Delivery: {msg.topic()}  offset={msg.offset()}")

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



class GraphStateManager:
    """
    Store and update graph state in Redis.

    Keys structure:
      graph:{entity_type}:{entity_id}:neighbors → adjacency list
      graph:{entity_type}:{entity_id}:features  → node features
      graph:emb:{entity_type}:{entity_id}       → cached embedding
      graph:cold_start:{entity_type}            → cold start embedding
    """

    ENTITY_TYPE_TO_CONTEXT_KEY = {
        "user": "user_id",
        "device": "device_id",
        "addr": "addr_id",
        "card": "card_id",
    }

    ENTITY_TYPE_TO_BATCH_KEY = {
        "user": "user_id",
        "device": "device_id",
        "addr": "addr_id",
        "card": "card_id",
    }

    def __init__(
            self,
            feature_store: FeatureStore,
            emb_dim: int,
            entity_feat_cols_dict):
        self.fs     = feature_store
        self.client = feature_store.client
        self.emb_dim = emb_dim
        self.entity_feat_cols_dict = entity_feat_cols_dict

    # Adjacency list
    def add_edge(
            self,
            entity_type: str,
            entity_id:   str,
            tx_id:       int | str,
            tx_dt:       int,
            tx_amt:      float,
    ) -> None:
        """
        Add edge: transaction → entity.
        Store last MAX_NEIGHBORS transactions for each entity.
        """
        key    = f"graph:{entity_type}:{entity_id}:neighbors"
        record = {
            "tx_id": tx_id,
            "dt":    tx_dt,
            "amt":   tx_amt,
        }
        self.client.lpush(key, json.dumps(record))
        self.client.ltrim(key, 0, MAX_NEIGHBORS - 1)
        self.client.expire(key, GRAPH_STATE_TTL)

    def get_neighbors(
            self,
            entity_type: str,
            entity_id:   str | None,
            k:           int = 10,
    ) -> list[dict]:
        """Return last k transactions for entity."""

        if not entity_id:
            return []
        key  = f"graph:{entity_type}:{entity_id}:neighbors"
        data = self.client.lrange(key, 0, k - 1)
        return [json.loads(d) for d in data]

    # Node features
    def update_node_features(
            self,
            entity_type: str,
            entity_id:   str,
            features:    dict,
    ) -> None:
        """Store updated node features."""
        key = f"graph:{entity_type}:{entity_id}:features"
        self.client.setex(key, GRAPH_STATE_TTL, json.dumps(features))

    def get_node_features(
            self,
            entity_type: str,
            entity_id:   str | None,
    ) -> dict | None:
        if not entity_id:
            return None
        key  = f"graph:{entity_type}:{entity_id}:features"
        data = self.client.get(key)
        return json.loads(data) if data else None

    def _init_entity_features(
        self,
        entity_type: str,
        tx_dt: int,
        tx_amt: float,
    ) -> dict:
        """
        Initializes online entity state.

        Internal state includes extra fields needed for online updates:
          tx_count, first_dt, last_dt, amt_sum, amt_m2, devices_seen, users_seen

        Output model features are created separately by _export_model_features().
        """
        return {
            "tx_count": 0,
            "first_dt": tx_dt,
            "last_dt": tx_dt,

            "amt_sum": 0.0,
            "amt_mean": tx_amt,
            "amt_std": 0.0,
            "amt_m2": 0.0,
            "amt_max": tx_amt,
            "amt_min": tx_amt,

            "devices_seen": [],
            "users_seen": [],

            # Online inference cannot know true future labels.
            # This should be replaced by train-fitted OOF/default fraud rate if available.
            "fraud_rate_oof": 0.0,
        }

    def _update_entity_features(
        self,
        entity_type: str,
        current: dict,
        tx: dict,
        graph_context: dict,
    ) -> dict:
        tx_dt = int(tx.get("TransactionDT", 0) or 0)
        tx_amt = float(tx.get("TransactionAmt", 0) or 0)

        old_count = int(current.get("tx_count", 0))
        new_count = old_count + 1

        old_mean = float(current.get("amt_mean", tx_amt))
        old_m2 = float(current.get("amt_m2", 0.0))

        new_mean = old_mean + (tx_amt - old_mean) / new_count
        new_m2 = old_m2 + (tx_amt - old_mean) * (tx_amt - new_mean)

        if new_count > 1:
            new_std = float((new_m2 / (new_count - 1)) ** 0.5)
        else:
            new_std = 0.0

        first_dt = int(current.get("first_dt", tx_dt) or tx_dt)
        last_dt = tx_dt

        devices_seen = set(current.get("devices_seen", []))
        users_seen = set(current.get("users_seen", []))

        device_id = graph_context.get("device_id")
        user_id = graph_context.get("user_id")

        if device_id:
            devices_seen.add(str(device_id))
        if user_id:
            users_seen.add(str(user_id))

        current.update({
            "tx_count": new_count,
            "first_dt": first_dt,
            "last_dt": last_dt,

            "amt_sum": round(float(current.get("amt_sum", 0.0)) + tx_amt, 4),
            "amt_mean": round(new_mean, 4),
            "amt_std": round(new_std, 4),
            "amt_m2": round(new_m2, 4),
            "amt_max": round(max(float(current.get("amt_max", tx_amt)), tx_amt), 4),
            "amt_min": round(min(float(current.get("amt_min", tx_amt)), tx_amt), 4),

            "devices_seen": list(devices_seen),
            "users_seen": list(users_seen),
        })

        return current

    def _export_model_features(
        self,
        entity_type: str,
        state: dict | None,
    ) -> dict:
        """
        Converts internal online state to batch-compatible entity feature schema.
        """
        batch_key = self.ENTITY_TYPE_TO_BATCH_KEY[entity_type]
        required_cols = self.entity_feat_cols_dict.get(batch_key, [])

        if state is None:
            state = {}

        tx_count = int(state.get("tx_count", 0))
        first_dt = state.get("first_dt")
        last_dt = state.get("last_dt")

        if first_dt is not None and last_dt is not None:
            time_span = (int(last_dt) - int(first_dt)) / 86400.0 + 1.0
        else:
            time_span = 1.0

        base = {
            "log_total_tx": float(np.log1p(tx_count)),
            "amt_mean": float(state.get("amt_mean", 0.0)),
            "amt_std": float(state.get("amt_std", 0.0)),
            "amt_max": float(state.get("amt_max", 0.0)),
            "amt_min": float(state.get("amt_min", 0.0)),
            "time_span": float(time_span),
            "fraud_rate_oof": float(state.get("fraud_rate_oof", 0.0)),

            "unique_devices_per_user": float(len(state.get("devices_seen", []))),
            "unique_users_per_device": float(len(state.get("users_seen", []))),
        }

        return {
            col: base.get(col, 0.0)
            for col in required_cols
        }
    

    # Embeddings cache
    def cache_embedding(
            self,
            entity_type: str,
            entity_id:   str,
            embedding:   np.ndarray,
    ) -> None:
        """
        Cache pre-computed embedding for entity.
        InkStream/Ripple approach: unchanged nodes → from cache,
        new/updated nodes → recompute.
        """
        key  = f"graph:emb:{entity_type}:{entity_id}"
        data = embedding.astype(np.float32).tobytes()
        self.client.setex(key, EMBEDDING_TTL, data)

    def get_cached_embedding(
            self,
            entity_type: str,
            entity_id:   str | None,
    ) -> np.ndarray | None:
        """Return cached embedding or None."""

        if not entity_id:
            return None
        
        key  = f"graph:emb:{entity_type}:{entity_id}"
        data = self.client.get(key)
        if not data:
            return None
        emb = np.frombuffer(data, dtype=np.float32).copy()

        if emb.shape[0] != self.emb_dim:
            return None
        return emb.tolist()

    def save_cold_start_embedding(
            self,
            entity_type: str,
            embedding:   np.ndarray,
    ) -> None:
        """Зберігає cold start embedding (mean з train)."""
        key  = f"graph:cold_start:{entity_type}"
        data = embedding.astype(np.float32).tobytes()
        self.client.set(key, data)

    def get_cold_start_embedding(
            self,
            entity_type: str,
    ) -> list[float]:
        """Return cold start embedding or zeros."""
        key  = f"graph:cold_start:{entity_type}"
        data = self.client.get(key)
        if data:
            emb = np.frombuffer(data, dtype=np.float32).copy()
            if emb.shape[0] == self.emb_dim:
                return emb.tolist()
        return np.zeros(self.emb_dim, dtype=np.float32).tolist()

    # Affected subgraph
    def get_affected_subgraph(
            self,
            graph_context: dict,
            k_neighbors:   int = 10,
    ) -> dict:
        """
        InkStream approach: define affected subgraph (2-hop).
        Return context for inference_consumer.
        """
        # user_id   = graph_context.get("user_id")
        # device_id = graph_context.get("device_id")
        # card_id   = graph_context.get("card_id")
        # addr_id   = graph_context.get("addr_id")

        entity_ids = {
            entity_type: graph_context.get(context_key)
            for entity_type, context_key in self.ENTITY_TYPE_TO_CONTEXT_KEY.items()
        }

        subgraph = {
            "neighbors": {},
            "node_features": {},
            "model_node_features": {},
            "cached_embeddings": {},
            "cold_start_embeddings": {},
            "affected_entities": {},
        }

        for entity_type, entity_id in entity_ids.items():
            state = self.get_node_features(entity_type, entity_id)

            subgraph["neighbors"][entity_type] = self.get_neighbors(
                entity_type,
                entity_id,
                k_neighbors,
            )
            subgraph["node_features"][entity_type] = state
            subgraph["model_node_features"][entity_type] = self._export_model_features(
                entity_type,
                state,
            )
            subgraph["cached_embeddings"][entity_type] = self.get_cached_embedding(
                entity_type,
                entity_id,
            )
            subgraph["cold_start_embeddings"][entity_type] = self.get_cold_start_embedding(
                entity_type,
            )
            subgraph["affected_entities"][entity_type] = entity_id

        return subgraph


    def update_graph_state(
            self,
            tx:            dict,
            graph_context: dict,
    ) -> None:
        """
        Update graph after new transaction:
        Add edges (tx → entity)
        Update node features
        """
        tx_id  = tx.get("TransactionID")
        tx_dt  = int(tx.get("TransactionDT",  0) or 0)
        tx_amt = float(tx.get("TransactionAmt", 0) or 0)

        entity_ids = {
            entity_type: graph_context.get(context_key)
            for entity_type, context_key in self.ENTITY_TYPE_TO_CONTEXT_KEY.items()
        }

        for entity_type, entity_id in entity_ids.items():
            if not entity_id:
                continue

            # Add edge
            self.add_edge(
                entity_type = entity_type, 
                entity_id   = entity_id, 
                tx_id       = tx_id, 
                tx_dt       = tx_dt, 
                tx_amt      = tx_amt,)

            # Update node features
            current = self.get_node_features(entity_type, entity_id) 
            if current is None:
                current = self._init_entity_features(
                    entity_type=entity_type,
                    tx_dt=tx_dt,
                    tx_amt=tx_amt,
                )

            current = self._update_entity_features(
                entity_type=entity_type,
                current=current,
                tx=tx,
                graph_context=graph_context,
            )

            self.update_node_features(
                entity_type=entity_type,
                entity_id=entity_id,
                features=current,
            )


def main():
    graph_state = GraphStateManager(
        feature_store         = feature_store,
        emb_dim               = EMB_DIM,
        entity_feat_cols_dict = ENTITY_FEAT_COLS_DICT,)

    consumer.subscribe([TOPIC_IN])
    print(f"Graph State Manager started")
    print(f"   Listening:  {TOPIC_IN}")
    print(f"   Forwarding: {TOPIC_OUT}\n")
    print(f"   Errors:     {TOPIC_ERRORS}")
    # print(f"   EMB_DIM:    {EMB_DIM}\n")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"[ERROR] Kafka consumer: {msg.error()}")
                continue
            
            payload = None
            try:
                payload       = json.loads(msg.value())
                tx            = payload.get("raw_tx", {})
                graph_context = payload.get("graph_context", {})
                tx_id         = payload.get("TransactionID")

                print(f"TransactionID={tx_id}")

                # Define affected subgraph
                subgraph = graph_state.get_affected_subgraph(graph_context)

                # Update graph
                graph_state.update_graph_state(
                    tx            = tx, 
                    graph_context = graph_context)

                # Send to graph.updates
                out = {
                    "TransactionID":    tx_id,
                    "tabular_features": payload.get("tabular_features", {}),
                    "graph_context":    graph_context,
                    "subgraph":         subgraph,   # for inference consumer
                    "raw_tx":           tx,
                }

                producer.produce(
                    topic    = TOPIC_OUT,
                    key      = str(tx_id),
                    value    = json.dumps(out, default=str),
                    callback = delivery_report,
                )
                producer.flush()
                consumer.commit(msg)
            except Exception as e:
                print(f"[ERROR] Graph state failed: {e}")
                send_error(
                    payload        = payload,
                    stage          = "graph_state_manager", 
                    error          = e,
                    original_topic = TOPIC_IN,)
                consumer.commit(msg)
                
    except KeyboardInterrupt:
        print("\n[WARN] Stopped by user")
    finally:
        consumer.close()
        print("[DONE] Consumer closed")


if __name__ == "__main__":
    main()