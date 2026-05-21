# kafka/entity_graph_store.py
#
# Завантажує batch-trained артефакти в Redis при старті:
#   val_to_idx mappings    → для entity node lookup
#   cold start embeddings  → для нових/невідомих entities
#   entity features        → pre-computed з train
#
# Запускати ОДИН РАЗ перед стартом inference_consumer

import json
import logging
import sys
import numpy as np
from pathlib import Path

PROJECT_DIR  = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_DIR / "ml" / "artifacts"
sys.path.append(str(PROJECT_DIR / "ml" / "src"))

import utils
from feature_store import FeatureStore

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [EGS] %(levelname)s %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

REDIS_PREFIX_VAL_TO_IDX    = "entity:val_to_idx"
REDIS_PREFIX_COLD_START    = "graph:cold_start"
REDIS_PREFIX_ENTITY_FEAT   = "entity:features"
TTL = 86400 * 7   # 7 днів


def load_entity_graph_store() -> None:
    """
    Load batch-trained artifacts in Redis.
    """
    feature_store = FeatureStore()
    client        = feature_store.client

    # Loading graph_config
    logger.info("Loading graph_config...")
    graph_config = utils.load_artifact(
        str(ARTIFACT_DIR / "hgnn_v3_baseline" / "configs" / "graph_config.pkl")
    )

    entity_cols         = graph_config.entity_cols
    val_to_idx_dict     = graph_config.val_to_idx_dict
    entity_features     = graph_config.entity_features
    entity_feat_cols    = graph_config.entity_feat_cols_dict

    logger.info(f"  entity_cols: {entity_cols}")

    # val_to_idx mappings
    # Потрібні для lookup entity node index під час inference
    logger.info("Loading val_to_idx mappings...")
    for col in entity_cols:
        if col not in val_to_idx_dict:
            logger.warning(f"  {col}: val_to_idx not found — skipping")
            continue

        val_to_idx = val_to_idx_dict[col]
        key        = f"{REDIS_PREFIX_VAL_TO_IDX}:{col}"

        # Save as JSON (entity_value → node_idx)
        client.setex(key, TTL, json.dumps(val_to_idx))
        logger.info(f"  {col}: {len(val_to_idx):,} mappings → {key}")

    # Cold start embeddings
    # Cold start = mean embedding from train for new entities
    logger.info("Loading cold start embeddings...")

    # Load from artifacts
    cold_start_path = ARTIFACT_DIR / "cold_start_embeddings.pkl"
    if cold_start_path.exists():
        cold_start_embs = utils.load_artifact(str(cold_start_path))
        for entity_type, emb in cold_start_embs.items():
            key  = f"{REDIS_PREFIX_COLD_START}:{entity_type}"
            data = np.array(emb, dtype=np.float32).tobytes()
            client.set(key, data)
            logger.info(f"  {entity_type}: cold_start emb dim={len(emb)} → {key}")
    else:
        # Fallback — zeros (буде замінено після першого HGNN inference)
        logger.warning(
            "cold_start_embeddings.pkl not found — "
            "using zeros (run save_cold_start_embeddings.py first)"
        )
        emb_dim = 64
        for entity_type in ["user", "device", "addr", "card"]:
            key  = f"{REDIS_PREFIX_COLD_START}:{entity_type}"
            data = np.zeros(emb_dim, dtype=np.float32).tobytes()
            client.set(key, data)
            logger.info(f"  {entity_type}: zeros({emb_dim}) → {key}")

    # Entity features (pre-computed from train)
    logger.info("Loading entity features...")
    for col in entity_cols:
        if col not in entity_features:
            logger.warning(f"  {col}: entity_features not found — skipping")
            continue

        ef       = entity_features[col]
        feat_cols = entity_feat_cols.get(col, [])
        val_to_idx = val_to_idx_dict.get(col, {})

        count = 0
        for _, row in ef.iterrows():
            entity_val = row[col]
            node_idx   = val_to_idx.get(entity_val)
            if node_idx is None:
                continue

            key      = f"{REDIS_PREFIX_ENTITY_FEAT}:{col}:{entity_val}"
            features = {
                c: float(row[c]) if c in row.index else 0.0
                for c in feat_cols
            }
            client.setex(key, TTL, json.dumps(features))
            count += 1

        logger.info(f"  {col}: {count:,} entity features loaded")

    logger.info("\nVerification:")
    for col in entity_cols:
        key   = f"{REDIS_PREFIX_VAL_TO_IDX}:{col}"
        count = len(json.loads(client.get(key) or "{}"))
        logger.info(f"  {key}: {count:,} entries")

    entity_keys = client.keys(f"{REDIS_PREFIX_ENTITY_FEAT}:*")
    logger.info(f"  entity:features:*: {len(entity_keys):,} keys")

    cold_keys = client.keys(f"{REDIS_PREFIX_COLD_START}:*")
    logger.info(f"  graph:cold_start:*: {len(cold_keys):,} keys")

    logger.info("Entity Graph Store loaded successfully")


if __name__ == "__main__":
    load_entity_graph_store()