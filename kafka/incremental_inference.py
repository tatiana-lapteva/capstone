# kafka/incremental_inference.py
#
# [TODO — Крок 13]
# Буде використовуватись в inference_consumer.py
#
# def build_micro_graph(subgraph, graph_context, graph_config) -> HeteroData:
#     """Будує micro-graph з Redis subgraph context."""
#
# def incremental_hgnn_inference(
#         tx:            dict,
#         subgraph:      dict,
#         graph_context: dict,
#         model:         HGNN,
#         graph_state:   GraphStateManager,
#         graph_config,
# ) -> float:
#     """
#     InkStream підхід:
#     1. Визначити affected nodes (вже в subgraph)
#     2. Незмінені вузли → cached embedding з Redis
#     3. Нові/змінені → forward pass тільки на micro-graph
#     4. Зберегти нові embeddings в Redis
#     """


# kafka/incremental_inference.py

from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData


SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


ENTITY_MAP = {
    "user": {
        "batch_node_type": "user_id",
        "context_key":     "user_id",
        "edge_name":       "to_user_id",
        "rev_edge_name":   "rev_user_id",
    },
    "device": {
        "batch_node_type": "device_id",
        "context_key":     "device_id",
        "edge_name":       "to_device_id",
        "rev_edge_name":   "rev_device_id",
    },
    "addr": {
        "batch_node_type": "addr_id",
        "context_key":     "addr_id",
        "edge_name":       "to_addr_id",
        "rev_edge_name":   "rev_addr_id",
    },
    "card": {
        "batch_node_type": "card_id",
        "context_key":     "card_id",
        "edge_name":       "to_card_id",
        "rev_edge_name":   "rev_card_id",
    },
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def get_graph_features(graph_config) -> list[str]:
    return list(getattr(graph_config, "graph_features", []))


def get_entity_feat_cols(graph_config, batch_node_type: str) -> list[str]:
    entity_feat_cols_dict = getattr(graph_config, "entity_feat_cols_dict", {})
    return list(entity_feat_cols_dict.get(batch_node_type, []))


def build_tx_feature_vector(tx: dict, graph_config) -> list[float]:
    """
    Builds current transaction feature vector using graph_config.graph_features.

    Important:
    tx should contain graph_features from feature-engineering stage.
    If a feature is missing, it is filled with 0.0.
    """
    graph_features = get_graph_features(graph_config)

    if not graph_features:
        return [
            safe_float(tx.get("TransactionAmt", 0.0)),
            safe_float(tx.get("TransactionDT", 0.0)),
        ]

    return [
        safe_float(tx.get(col, 0.0))
        for col in graph_features
    ]


def build_entity_feature_vector(
    subgraph: dict,
    entity_type: str,
    graph_config,
) -> list[float]:
    """
    Builds entity feature vector using batch-compatible entity feature schema.
    """
    batch_node_type = ENTITY_MAP[entity_type]["batch_node_type"]
    required_cols = get_entity_feat_cols(graph_config, batch_node_type)

    model_node_features = (
        subgraph
        .get("model_node_features", {})
        .get(entity_type, {})
    )

    if not required_cols:
        return [0.0]

    return [
        safe_float(model_node_features.get(col, 0.0))
        for col in required_cols
    ]


def build_neighbor_tx_features(
    neighbors: list[dict],
    graph_config,
) -> list[list[float]]:
    """
    Builds simplified transaction features for historical neighbor transactions.

    Neighbor records from Redis usually contain:
      tx_id, dt, amt

    Since they do not contain full graph_features, we create compatible
    fallback vectors with available values.
    """
    graph_features = get_graph_features(graph_config)
    n_features = len(graph_features) if graph_features else 2

    rows = []

    for item in neighbors:
        amt = safe_float(item.get("amt", item.get("TransactionAmt", 0.0)))
        dt = safe_float(item.get("dt", item.get("TransactionDT", 0.0)))

        row = [0.0] * n_features

        if graph_features:
            for i, col in enumerate(graph_features):
                if col == "TransactionAmt":
                    row[i] = amt
                elif col == "TransactionDT":
                    row[i] = dt
                elif col in ("log_dt_prev", "dt_prev"):
                    row[i] = 0.0
                elif col == "is_burst":
                    row[i] = 0.0
                else:
                    row[i] = 0.0
        else:
            row = [amt, dt]

        rows.append(row)

    return rows


def make_edge_attr(
    n_edges: int,
    graph_config,
    temporal_decay: float = 1.0,
    fraud_rate: float = 0.0,
    freq_weight: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor | None:
    """
    Builds edge_attr if graph_config.edge_attr_dim is used.
    """
    edge_dim = int(getattr(graph_config, "edge_attr_dim", 0) or 0)

    if edge_dim <= 0:
        return None

    base = [freq_weight, fraud_rate, temporal_decay]

    if edge_dim <= len(base):
        values = base[:edge_dim]
    else:
        values = base + [0.0] * (edge_dim - len(base))

    return torch.tensor(
        [values for _ in range(n_edges)],
        dtype=torch.float32,
        device=device,
    )


def build_micro_graph(
    tx: dict,
    subgraph: dict,
    graph_context: dict,
    graph_config,
    device: str = "cpu",
) -> HeteroData:
    """
    Builds a micro-graph for online HGNN embedding inference.

    Node types:
      - transaction
      - user_id
      - device_id
      - addr_id
      - card_id

    Current transaction index is always 0 in data["transaction"].x.
    Historical neighbor transactions are appended after it.
    """
    data = HeteroData()

    current_tx_x = build_tx_feature_vector(tx, graph_config)

    neighbor_rows = []
    neighbors_by_entity = subgraph.get("neighbors", {})

    for entity_type in ENTITY_MAP:
        neighbor_rows.extend(
            build_neighbor_tx_features(
                neighbors_by_entity.get(entity_type, []),
                graph_config,
            )
        )

    tx_rows = [current_tx_x] + neighbor_rows

    data["transaction"].x = torch.tensor(
        tx_rows,
        dtype=torch.float32,
        device=device,
    )

    # Entity nodes: one affected node per entity type
    for entity_type, meta in ENTITY_MAP.items():
        batch_node_type = meta["batch_node_type"]

        entity_x = build_entity_feature_vector(
            subgraph=subgraph,
            entity_type=entity_type,
            graph_config=graph_config,
        )

        data[batch_node_type].x = torch.tensor(
            [entity_x],
            dtype=torch.float32,
            device=device,
        )

        # Current transaction -> entity
        edge_index = torch.tensor(
            [[0], [0]],
            dtype=torch.long,
            device=device,
        )

        edge_type = ("transaction", meta["edge_name"], batch_node_type)
        rev_edge_type = (batch_node_type, meta["rev_edge_name"], "transaction")

        data[edge_type].edge_index = edge_index
        data[rev_edge_type].edge_index = edge_index.flip(0)

        edge_attr = make_edge_attr(
            n_edges=1,
            graph_config=graph_config,
            device=device,
        )

        if edge_attr is not None:
            data[edge_type].edge_attr = edge_attr
            data[rev_edge_type].edge_attr = edge_attr.clone()

    return data


def extract_transaction_embedding(model_output) -> np.ndarray:
    """
    Extracts semantic transaction embedding from different possible HGNN outputs.

    Supported outputs:
      1. Tensor [num_tx, emb_dim]
      2. Dict with keys:
           - "transaction"
           - "transaction_emb"
           - "tx_emb"
           - "embeddings"
      3. Tuple/list where first item is embedding tensor
    """

    output = model_output

    if isinstance(output, dict):
        for key in ["transaction", "transaction_emb", "tx_emb", "x_transaction"]:
            if key in output:
                emb = output[key]
                return tensor_to_numpy_first_row(emb)

        if "embeddings" in output:
            embeddings = output["embeddings"]

            if isinstance(embeddings, dict):
                for key in ["transaction", "transaction_emb", "tx_emb"]:
                    if key in embeddings:
                        return tensor_to_numpy_first_row(embeddings[key])

            return tensor_to_numpy_first_row(embeddings)

        raise ValueError(
            f"Cannot extract transaction embedding from model output keys: {list(output.keys())}"
        )

    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise ValueError("HGNN model returned empty tuple/list")
        return extract_transaction_embedding(output[0])

    return tensor_to_numpy_first_row(output)


def tensor_to_numpy_first_row(value) -> np.ndarray:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    value = value.detach().cpu()

    if value.ndim == 1:
        emb = value.numpy()
    elif value.ndim == 2:
        emb = value[0].numpy()
    else:
        raise ValueError(f"Unsupported embedding tensor shape: {tuple(value.shape)}")

    emb = emb.astype(np.float32)

    if np.isnan(emb).any():
        raise ValueError("NaN detected in HGNN embedding")

    if np.isinf(emb).any():
        raise ValueError("Inf detected in HGNN embedding")

    return emb


def run_hgnn_model(model, micro_graph: HeteroData):
    """
    Runs HGNN model using the most common forward signatures.
    HGNN.forward(x_dict, edge_index_dict, edge_attr_dict)
    Return (logits, x_semantic, x_graph)
    """
    edge_attr_dict = {
        et: micro_graph[et].edge_attr
        for et in micro_graph.edge_types
        if hasattr(micro_graph[et], "edge_attr")
        and micro_graph[et].edge_attr is not None
    }
    # ✅ Відповідає HGNN.forward підпису
    output = model(
        micro_graph.x_dict,
        micro_graph.edge_index_dict,
        edge_attr_dict or None,
    )
    return output


def cache_entity_embeddings_if_available(
    model_output,
    graph_context: dict,
    graph_state,
) -> None:
    """
    Caches entity embeddings if HGNN output provides them.

    This is optional. If the model does not return entity embeddings,
    the function silently does nothing.
    """
    if not isinstance(model_output, dict):
        return

    embeddings = model_output.get("entity_embeddings") or model_output.get("embeddings")

    if not isinstance(embeddings, dict):
        return

    for entity_type, meta in ENTITY_MAP.items():
        entity_id = graph_context.get(meta["context_key"])
        batch_node_type = meta["batch_node_type"]

        if not entity_id:
            continue

        if batch_node_type in embeddings:
            emb = tensor_to_numpy_first_row(embeddings[batch_node_type])
        elif entity_type in embeddings:
            emb = tensor_to_numpy_first_row(embeddings[entity_type])
        else:
            continue

        graph_state.cache_embedding(
            entity_type=entity_type,
            entity_id=entity_id,
            embedding=emb,
        )


def extract_transaction_embedding(model_output) -> np.ndarray:
    """
    HGNN.forward повертає (logits, x_semantic, x_graph).
    Для XGBoost використовуємо x_semantic (index=1).
    """
    if isinstance(model_output, tuple):
        if len(model_output) == 3:
            # (logits, x_semantic, x_graph) — наш HGNN
            _, x_semantic, _ = model_output
            return tensor_to_numpy_first_row(x_semantic)
        if len(model_output) == 2:
            # (logits, x_semantic)
            _, x_semantic = model_output
            return tensor_to_numpy_first_row(x_semantic)

    # Fallback
    return tensor_to_numpy_first_row(model_output)


def incremental_hgnn_embedding(
    tx: dict,
    subgraph: dict,
    graph_context: dict,
    model,
    graph_state,
    graph_config,
    device: str = "cpu",
) -> np.ndarray:
    """
    InkStream/Ripple-style online HGNN embedding inference.

    Steps:
      1. Build affected micro-graph from Redis graph state.
      2. Reuse unchanged node context from subgraph.
      3. Run HGNN only on local micro-graph.
      4. Extract semantic transaction embedding.
      5. Cache entity embeddings if available.

    Returns:
      np.ndarray — semantic transaction embedding for XGBoost.
    """
    model.eval()

    micro_graph = build_micro_graph(
        tx=tx,
        subgraph=subgraph,
        graph_context=graph_context,
        graph_config=graph_config,
        device=device,
    )

    with torch.no_grad():
        output = run_hgnn_model(model, micro_graph)

    embedding = extract_transaction_embedding(output)

    cache_entity_embeddings_if_available(
        model_output=output,
        graph_context=graph_context,
        graph_state=graph_state,
    )

    return embedding