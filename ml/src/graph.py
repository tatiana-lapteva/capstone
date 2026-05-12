import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
from sklearn.metrics import roc_auc_score, average_precision_score   
from scipy.special import expit  # sigmoid
from sklearn.model_selection import TimeSeriesSplit
import utils
from torch_geometric.sampler import NegativeSampling
from torch_geometric.utils import structured_negative_sampling
from contextlib import contextmanager

@contextmanager
def eval_mode(model: torch.nn.Module):
    """
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)   

        
### =============== ENTITY FEATURES ===============

def compute_fraud_rate_oof(
        df:       pd.DataFrame,
        col:      str,
        n_splits: int = 5,
) -> np.ndarray:
    global_mean = float(df["isFraud"].mean())
    oof  = np.full(len(df), global_mean, dtype=np.float64)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, val_idx in tscv.split(df):
        fold_train = df.iloc[train_idx]
        fold_mean  = float(fold_train["isFraud"].mean())
        fold_map   = fold_train.groupby(col)["isFraud"].mean()
        oof[val_idx] = (
            df.iloc[val_idx][col]
            .map(fold_map)
            .fillna(fold_mean)
            .values
        )
    return oof


def build_entity_features(
        df:               pd.DataFrame, 
        col:              str, 
        entity_feat_cols: list, 
        compute_oof: bool = True) -> pd.DataFrame:
    grp      = df.groupby(col)
    features = pd.DataFrame({col: grp.size().index})

    features["log_total_tx"] = np.log1p(grp.size().values)
    features["amt_mean"]     = grp["TransactionAmt"].mean().values
    features["amt_std"]      = grp["TransactionAmt"].std().fillna(0).values
    features["time_span"]    = (
        grp["TransactionDT"].max() - grp["TransactionDT"].min()
    ).values

    features["amt_max"] = grp["TransactionAmt"].max().values
    features["amt_min"] = grp["TransactionAmt"].min().values
    features["amt_cv"]  = (
        features["amt_std"] / (features["amt_mean"] + 1e-8)
    )

    time_span_days = features["time_span"] / 86400.0
    features["tx_per_day"] = (
        grp.size().values / (time_span_days + 1.0)  
    )

    if "hour_sin" in df.columns or "TransactionDT" in df.columns:
        df_tmp         = df.copy()
        df_tmp["hour"] = (df_tmp["TransactionDT"] % 86400) // 3600
        df_tmp["is_night"] = df_tmp["hour"].between(0, 5).astype(int)
        night_ratio    = df_tmp.groupby(col)["is_night"].mean()
        features["night_tx_ratio"] = (
            night_ratio.reindex(features[col]).fillna(0).values
        )
    else:
        features["night_tx_ratio"] = 0.0

    if "is_burst" in df.columns:
        burst_ratio = df.groupby(col)["is_burst"].mean()
        features["burst_ratio"] = (
            burst_ratio.reindex(features[col]).fillna(0).values
        )
    else:
        features["burst_ratio"] = 0.0

    # Cross-groupby
    if "device_id" in df.columns and col != "device_id":
        # Number of unique device per every entity
        device_per_entity = df.groupby(col)["device_id"].nunique()
        features["unique_devices_per_user"] = (
            device_per_entity
            .reindex(features[col])
            .fillna(1)
            .values
        )

        # Number of unique entity per every device
        entity_per_device = df.groupby("device_id")[col].nunique()
        avg_users_per_device = (
            df.groupby(col)["device_id"]
            .apply(lambda devs: entity_per_device.reindex(devs).mean())
            .fillna(1)
        )
        features["unique_users_per_device"] = (
            avg_users_per_device
            .reindex(features[col])
            .fillna(1)
            .values
        )
    else:
        features["unique_devices_per_user"] = 1.0
        features["unique_users_per_device"] = 1.0

    # Fraud device ratio
    if "device_id" in df.columns and col != "device_id" and "isFraud" in df.columns:
        def fraud_device_ratio(g):
            total_devices = g["device_id"].nunique()
            if total_devices == 0:
                return 0.0
            fraud_devices = g[g["isFraud"] == 1]["device_id"].nunique()
            return fraud_devices / total_devices
        fdr = grp.apply(fraud_device_ratio)
        features["device_reuse_ratio"] = (
            fdr
            .reindex(features[col])
            .fillna(0)
            .values
        )
    else:
        features["device_reuse_ratio"] = 0.0

    # OOF fraud rate
    if compute_oof and "isFraud" in df.columns:
        oof = compute_fraud_rate_oof(df, col)
        oof_per_entity = (
            pd.Series(oof, index=df.index)
            .groupby(df[col])
            .mean()
        )
        features["fraud_rate_oof"] = (
            oof_per_entity
            .reindex(features[col])
            .fillna(0)
            .values
        )
    else:
        features["fraud_rate_oof"] = 0.0

    # Columns Validation
    for c in entity_feat_cols:
        if c not in features.columns:
            features[c] = 0.0

    return features[[col] + entity_feat_cols]


def build_card_similarity_edges(
        full_df:           pd.DataFrame,
        group_col:         str   = "card_id",
        amount_col:        str   = "amt_vs_user_mean",
        time_col:          str   = "TransactionDT",
        max_neighbors:     int   = 3,
        max_group_size:    int   = 500,
        max_time_diff:     int   = 86400,
        amt_sim_threshold: float = 0.5,
        global_fraud_rate: float = 0.035,
        train_df:          pd.DataFrame = None,
        sort_by_amount:    bool  = False,
) -> tuple:
    N = len(full_df)

    if train_df is not None and group_col in train_df.columns:
        fraud_rate_map = train_df.groupby(group_col)["isFraud"].mean()
    else:
        fraud_rate_map = pd.Series(dtype=float)

    full_df_tmp             = full_df.copy()
    full_df_tmp["orig_idx"] = np.arange(N)

    full_sorted = full_df_tmp.sort_values(
        [group_col, amount_col] if sort_by_amount else time_col
    )

    is_binary = amt_sim_threshold == 0.0
    edges_src, edges_dst, edge_attrs = [], [], []

    for group_val, group in full_sorted.groupby(group_col, sort=False):
        n = len(group)
        if n < 2:
            continue

        # recent/largest most important
        if n > max_group_size:
            group = group.tail(max_group_size)
            n     = max_group_size

        times     = group[time_col].values.astype(np.float32)
        amts      = group[amount_col].fillna(0.0).values.astype(np.float32)
        orig_idxs = group["orig_idx"].values

        raw_fraud = float(fraud_rate_map.get(group_val, global_fraud_rate))
        fraud_w   = float(np.clip(
            np.log1p(raw_fraud / (global_fraud_rate + 1e-8)),
            0.0, 3.0
        ))

        # Vectorized similarity matrix
        if is_binary:
            # binary: sim=1 
            sim_matrix = (amts[:, None] == amts[None, :]).astype(np.float32)
        else:
            denom_matrix = np.maximum(
                np.abs(amts[:, None]),
                np.abs(amts[None, :]),
            )
            denom_matrix = np.maximum(denom_matrix, 1e-8)
            sim_matrix   = 1.0 - np.abs(amts[:, None] - amts[None, :]) / denom_matrix
            sim_matrix   = np.clip(sim_matrix, 0.0, 1.0)

        # Time filter matrix — (n, n) bool
        dt_matrix = np.abs(times[:, None] - times[None, :])

        if sort_by_amount:
            # time as additional filter
            time_ok = dt_matrix <= max_time_diff
        else:
            # time як base filter
            time_ok = dt_matrix <= max_time_diff

        valid_mask = (
            time_ok
            & (sim_matrix >= amt_sim_threshold)
            & ~np.eye(n, dtype=bool)
        )

        # Temporal decay matrix
        decay_matrix = np.exp(-dt_matrix / max(max_time_diff, 1))

        # Top-K for each node
        for i in range(n):
            valid_j = np.where(valid_mask[i])[0]

            if len(valid_j) == 0:
                continue

            # Top-K for similarity
            if len(valid_j) > max_neighbors:
                top_k_local = np.argpartition(
                    -sim_matrix[i, valid_j], max_neighbors
                )[:max_neighbors]
                valid_j = valid_j[top_k_local]

            for j in valid_j:
                src = int(orig_idxs[i])
                dst = int(orig_idxs[j])
                sim_w   = float(sim_matrix[i, j])
                t_decay = float(decay_matrix[i, j])

                edges_src.extend([src, dst])
                edges_dst.extend([dst, src])
                edge_attrs.extend([
                    [sim_w, fraud_w, t_decay],
                    [sim_w, fraud_w, t_decay],
                ])

    if not edges_src:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float32),
        )

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attrs,              dtype=torch.float32)

    assert edge_index.max() < N
    assert edge_index.min() >= 0
    assert edge_attr.shape[1] == 3
    assert not torch.isnan(edge_attr).any()

    return edge_index, edge_attr


### =============== GRAPH BUILDER ===============
def build_graph(                        # weighted
        train_df:        pd.DataFrame,
        val_df:          pd.DataFrame,
        graph_config,
) -> tuple:
    """
    """
    graph_features   = graph_config.graph_features
    entity_cols      = graph_config.entity_cols
    
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    N       = len(full_df)
    n_train = len(train_df)

    assert N == len(train_df) + len(val_df), \
        f"N mismatch: {N} != {len(train_df)} + {len(val_df)}"
    print(f"✓ Full dataset: {N:,} tx ({n_train:,} train + {N - n_train:,} val)")

    data = HeteroData()

    # Masks
    train_mask                     = torch.zeros(N, dtype=torch.bool)
    train_mask[:n_train]           = True
    data["transaction"].train_mask = train_mask
    data["transaction"].val_mask   = ~train_mask

    assert data["transaction"].train_mask.sum() == n_train,     "Train mask wrong!"
    assert data["transaction"].val_mask.sum()   == N - n_train, "Val mask wrong!"
    # print(f"Masks: train={train_mask.sum():,}  val={(~train_mask).sum():,}")

    # Transaction node features 
    missing_feats = [f for f in graph_features if f not in full_df.columns]
    assert not missing_feats, f"Missing GRAPH_FEATURES: {missing_feats}"

    feat = full_df[graph_features].fillna(0).values.astype(np.float32)

    if graph_config.feat_scaler is None:
        graph_config.feat_scaler = StandardScaler()
        graph_config.feat_scaler.fit(feat[:n_train])   # ← fit тільки на train
    feat = graph_config.feat_scaler.transform(feat)

    assert not np.isnan(feat).any(), "NaN in transaction features!"
    assert not np.isinf(feat).any(), "Inf in transaction features!"

    data["transaction"].x = torch.tensor(feat, dtype=torch.float32)
    data["transaction"].y = torch.tensor(
        full_df["isFraud"].values, dtype=torch.float32
    )

    print(f"✓ Transaction x: {data['transaction'].x.shape}")
    print(f"✓ Fraud ratio  : {data['transaction'].y.mean().item():.4f}")

    global_fraud_rate = float(train_df["isFraud"].mean())
    num_nodes_dict = {}
    total_edges    = 0

    # Entity features
    for col in entity_cols:
        # Separate features for each entity type
        if col not in full_df.columns:
            print(f"{col} not in full_df — skipping")
            continue
        feat_cols = graph_config.entity_feat_cols_dict.get(
            col,
            # graph_config.entity_feat_cols  # fallback
        )
        feat_dim = len(feat_cols)

        # Val to idx mapping
        if col not in graph_config.val_to_idx_dict:
            train_vals = train_df[col].dropna().unique()
            val_to_idx = {val: idx + 1 for idx, val in enumerate(train_vals)}
            graph_config.val_to_idx_dict[col] = val_to_idx
            print(f"  ✓ {col}: val_to_idx fitted ({len(train_vals):,} unique values)")
        else:
            val_to_idx = graph_config.val_to_idx_dict[col]
            print(f"  ✓ {col}: val_to_idx reused from graph_config")    
            
        num_nodes           = len(val_to_idx) + 1
        num_nodes_dict[col] = num_nodes
        
        # Cold start node 0 for unknown
        dst_values = full_df[col].map(val_to_idx).fillna(0).astype(int)
        n_unknown  = (dst_values == 0).sum()

        if n_unknown > 0:
            print(f"{col}: {n_unknown:,} unknown → cold start node 0")

        dst_tensor = torch.tensor(dst_values.values, dtype=torch.long)
        src_tensor = torch.tensor(np.arange(N),       dtype=torch.long)

        if col not in graph_config.entity_features:
            ef = build_entity_features(train_df, col, feat_cols)
            graph_config.entity_features[col] = ef
        else:
            ef = graph_config.entity_features[col]
            print(f"{col}: entity_features reused from graph_config")
            cached_cols = set(ef.columns) - {col}
            expected_cols = set(feat_cols)
            if cached_cols != expected_cols:
                print(f"{col}: feat_cols змінились → rebuild")
                ef = build_entity_features(train_df, col, feat_cols)
                graph_config.entity_features[col] = ef
            else:
                print(f"{col}: entity_features reused from graph_config")

        # Align features by node index 
        feat_arr = np.zeros((num_nodes, feat_dim), dtype=np.float32)
        for _, row in ef.iterrows():
            node_idx = val_to_idx.get(row[col])
            if node_idx is not None and 0 < node_idx < num_nodes:
                feat_arr[node_idx] = row[feat_cols].values.astype(np.float32)

        assert not np.isnan(feat_arr).any(), f"NaN in {col} entity features!"

        # Cold start node = mean of known nodes
        known_feats = feat_arr[1:]
        nonzero_mask  = np.any(known_feats != 0, axis=1)
        if nonzero_mask.sum() > 0:
            feat_arr[0] = known_feats[nonzero_mask].mean(axis=0)
            print(f"{col}: cold start node = mean of "
                f"{nonzero_mask.sum():,} known entities")

        assert not np.isnan(feat_arr).any(), f"NaN in {col} entity features!"



        # Column-based StandardScaler
        mask_nonzero = np.any(feat_arr != 0, axis=1)

        if col not in graph_config.entity_scalers:
            if mask_nonzero.sum() > 1:
                scaler = StandardScaler()
                scaler.fit(feat_arr[mask_nonzero])
                graph_config.entity_scalers[col] = scaler
            else:
                graph_config.entity_scalers[col] = None
                print(f"{col}: too few non-zero nodes — scaler skipped")
        
        scaler = graph_config.entity_scalers[col]
        if scaler is not None:
            feat_arr_scaled              = feat_arr.copy()
            feat_arr_scaled[mask_nonzero] = scaler.transform(feat_arr[mask_nonzero])
        else:
            feat_arr_scaled = feat_arr

        assert not np.isnan(feat_arr_scaled).any(), \
            f"NaN in {col} entity features after scaling!"

        data[col].x = torch.tensor(feat_arr_scaled, dtype=torch.float32)

        # Freq weight
        freq_map  = train_df.groupby(col).size()
        fraud_rate_map = train_df.groupby(col)["isFraud"].mean()
        freq_vals = (
        full_df[col]
        .map(1.0 / freq_map)
        .fillna(1.0 / freq_map.max())   # unknown → min frequency
        .values.astype(np.float32)
    )
        freq_vals = freq_vals / (freq_vals.max() + 1e-8)  # normalize [0,1]

        # Fraud_rate weight  
        fraud_rate_vals = (
            full_df[col]
            .map(fraud_rate_map)
            .fillna(global_fraud_rate)       # unknown → global mean
            .values.astype(np.float32)
            )
        fraud_rate_vals = np.clip(fraud_rate_vals, 0.0, 1.0)
        temporal_vals = np.ones(N, dtype=np.float32)

        ew_np = np.column_stack([freq_vals, fraud_rate_vals, temporal_vals])
        ew = torch.tensor(ew_np, dtype=torch.float32)

        assert ew.shape == (N, 3)
        assert not np.isnan(ew_np).any()
        assert src_tensor.max() < N,         f"{col}: src out of range"
        assert dst_tensor.max() < num_nodes, f"{col}: dst out of range"

        data["transaction", f"to_{col}",  col].edge_index = torch.stack([src_tensor, dst_tensor])
        data["transaction", f"to_{col}",  col].edge_attr  = ew
        data[col, f"rev_{col}", "transaction"].edge_index = torch.stack([dst_tensor, src_tensor])
        data[col, f"rev_{col}", "transaction"].edge_attr  = ew

        n_edges      = N
        total_edges += n_edges * 2
        unknown_pct  = n_unknown / N * 100
        print(f"  ✓ {col}: {num_nodes:,} nodes  {n_edges:,} edges/dir  "
              f"feat_dim={feat_dim}  "
              f"fraud_rate_mean={fraud_rate_vals.mean():.4f}  "
              f"unknown={unknown_pct:.1f}%")

    assert total_edges > 0, "NO ENTITY EDGES — check ENTITY_COLS!"
    print(f"Total entity edges: {total_edges:,}")

    graph_config.num_nodes_dict = num_nodes_dict

    # Temporal edges 
    if graph_config.temporal_enabled:
        full_df_tmp             = full_df.copy()
        full_df_tmp["orig_idx"] = np.arange(N)
        full_sorted             = full_df_tmp.sort_values("TransactionDT")
        max_k = graph_config.max_temporal_neighbors

        for group_col in graph_config.temporal_group_cols:
            if group_col not in full_sorted.columns:
                print(f"{group_col} not in dataset — skipping")
                continue
            threshold = (
                graph_config.temporal_thresholds.get(group_col)
                if hasattr(graph_config, "temporal_thresholds")
                else None
            ) or graph_config.temporal_threshold

            edges_src, edges_dst, t_weights = [], [], []

            freq_map_col       = train_df.groupby(group_col).size()
            fraud_rate_map_col = train_df.groupby(group_col)["isFraud"].mean()
            min_freq_col       = freq_map_col.min()
            for group_val, group in full_sorted.groupby(group_col):
                times     = group["TransactionDT"].values
                orig_idxs = group["orig_idx"].values
                n         = len(times)

                if n < 2:
                    continue
                
                raw_freq  = freq_map_col.get(group_val, 1)
                freq_w    = float((1.0 / raw_freq) / (1.0 / min_freq_col + 1e-8))
                freq_w    = min(freq_w, 1.0)
                fraud_w   = float(fraud_rate_map_col.get(group_val, global_fraud_rate))
                fraud_w   = float(np.clip(fraud_w, 0.0, 1.0))

                for i in range(n):
                    upper = min(i + 1 + max_k, n)
                    for j in range(i + 1, upper):
                        dt = times[j] - times[i]
                        if dt > threshold:
                            break  
                        temporal_decay = float(np.exp(-dt / threshold))
                        edges_src.append(orig_idxs[i])
                        edges_dst.append(orig_idxs[j])
                        t_weights.append([freq_w, fraud_w, temporal_decay])
            edge_type = f"temporal_{group_col}"

            if edges_src:
                edge_index = torch.tensor(
                    [edges_src, edges_dst], dtype=torch.long
                )
                edge_attr = torch.tensor(
                    t_weights, dtype=torch.float32)    # .unsqueeze(1)

                assert edge_attr.shape[1] == 3, \
                    f"{edge_type}: edge_attr shape {edge_attr.shape}"
                assert edge_index.max() < N, \
                    f"{edge_type}: edge index {edge_index.max()} >= N={N}"
                assert edge_index.min() >= 0, \
                    f"{edge_type}: negative edge index"

                data["transaction", edge_type, "transaction"].edge_index = edge_index
                data["transaction", edge_type, "transaction"].edge_attr  = edge_attr

                print(f"  ✓ {edge_type}: {edge_index.size(1):,} edges  "
                    f"avg_decay={edge_attr[:,2].mean():.4f}  "
                    f"avg_fraud={edge_attr[:,1].mean():.4f}  "
                    f"threshold={int(threshold)//3600}h")
            else:
                data["transaction", edge_type, "transaction"].edge_index = \
                    torch.empty((2, 0), dtype=torch.long)
                data["transaction", edge_type, "transaction"].edge_attr  = \
                    torch.empty((0, 3), dtype=torch.float32)
                print(f"  ✓ {edge_type}: empty  threshold={int(threshold)//3600}h")

    else:
        for group_col in graph_config.temporal_group_cols:
            edge_type = f"temporal_{group_col}"
            data["transaction", edge_type, "transaction"].edge_index = \
                torch.empty((2, 0), dtype=torch.long)
            data["transaction", edge_type, "transaction"].edge_attr  = \
                torch.empty((0, 3), dtype=torch.float32)
        print("Temporal edges: disabled")

    if graph_config.similarity_enabled:
        for sim_cfg in graph_config.similarity_configs:
            feat_col  = sim_cfg.get("amount_col", "amt_vs_user_mean")
            edge_type = sim_cfg["edge_type"]

            if feat_col not in full_df.columns:
                print(f"{feat_col} not in full_df — skipping {edge_type}")
                data["transaction", edge_type, "transaction"].edge_index = \
                    torch.empty((2, 0), dtype=torch.long)
                data["transaction", edge_type, "transaction"].edge_attr  = \
                    torch.empty((0, 3), dtype=torch.float32)
                continue

            edge_index, edge_attr = build_card_similarity_edges(
                full_df           = full_df,
                group_col         = sim_cfg.get("group_col", "card_id"),
                amount_col        = feat_col,
                time_col          = sim_cfg.get("time_col", "TransactionDT"),
                max_neighbors     = sim_cfg.get("max_neighbors", 3),
                max_group_size    = sim_cfg.get("max_group_size", 500),
                max_time_diff     = sim_cfg.get("max_time_diff", 86400),
                amt_sim_threshold = sim_cfg.get("amt_sim_threshold", 0.5),
                global_fraud_rate = global_fraud_rate,
                train_df          = train_df,
                sort_by_amount    = sim_cfg.get("sort_by_amount", False),
            )

            data["transaction", edge_type, "transaction"].edge_index = edge_index
            data["transaction", edge_type, "transaction"].edge_attr  = edge_attr

            if edge_index.size(1) > 0:
                print(f"{edge_type}: {edge_index.size(1):,} edges  "
                    f"avg_sim={edge_attr[:,0].mean():.4f}  "
                    f"avg_fraud={edge_attr[:,1].mean():.4f}")
            else:
                print(f"{edge_type}: empty")

    assert data["transaction"].num_nodes > 0, "Empty transaction nodes!"
    assert len(data.edge_types) > 0,          "No edge types!"

    for et in data.edge_types:
        ei = data[et].edge_index
        assert ei.size(0) == 2, f"{et}: wrong edge_index shape"
        if ei.size(1) > 0:  
            assert ei.max() < N,  f"{et}: edge index out of range"
            assert ei.min() >= 0, f"{et}: negative edge index"

    print("Graph built successfully")
    print(f"  Node types : {list(data.node_types)}")
    print(f"  Edge types : {len(data.edge_types)}")
    print(f"  Total nodes: {sum(data[n].num_nodes for n in data.node_types):,}")

    return data


### =============== LOADER ===============

def _get_mask(data, mask_type: str) -> torch.Tensor:
    if mask_type == "train":
        return data["transaction"].train_mask
    elif mask_type == "val":
        return data["transaction"].val_mask
    return torch.ones(data["transaction"].num_nodes, dtype=torch.bool)


def get_loader(
        data, 
        mask_type: str = "train", 
        config         = None,            #    TrainConfig 
        y_train        = None, 
        seed:      int = 42) -> NeighborLoader:
    
    batch_size    = config.batch_size_train if mask_type == "train" else config.batch_size_val
    mask          = _get_mask(data, mask_type)
    indices       = mask.nonzero(as_tuple=True)[0]
    num_neighbors = {key: config.num_neighbors for key in data.edge_types}

    if mask_type == "train" and config.use_imbalanced_sampler and y_train is not None:
        labels  = torch.tensor(y_train.values, dtype=torch.long)
        torch.manual_seed(seed)
        sampler = ImbalancedSampler(labels)
        return NeighborLoader(
            data,
            num_neighbors = num_neighbors,
            batch_size    = batch_size,
            input_nodes   = ("transaction", indices),
            sampler       = sampler,
            shuffle       = False,
        )
    return NeighborLoader(
        data,
        num_neighbors = num_neighbors,
        batch_size    = batch_size,
        input_nodes   = ("transaction", indices),
        shuffle       = (mask_type == "train"),
    )
    


### =============== FOCAL LOSS ===============

class FocalLoss(nn.Module):
    def __init__(
            self, 
            alpha:      float = 0.25, 
            gamma:      float = 2.0, 
            pos_weight: float = 1.0, 
            reduction:  str = "mean"
            ):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.pos_weight = pos_weight
        self.reduction  = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss     = F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight = torch.tensor(self.pos_weight, device=logits.device),
            reduction  = "none",
            )
        pt          = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss    = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
    

class FraudContrastiveLoss(nn.Module):
    """
    """
    def __init__(self, margin: float = 0.3, temperature: float = 0.1):
        super().__init__()
        self.margin      = margin
        self.temperature = temperature

    def forward(
            self,
            emb:      torch.Tensor,
            targets:  torch.Tensor,
            batch_sz: int,
    ) -> torch.Tensor:
        emb_seed = F.normalize(emb[:batch_sz], dim=1)
        y        = targets.bool()

        n_pos = y.sum().item()
        n_neg = (~y).sum().item()

        if n_pos < 2 or n_neg < 2:
            return torch.tensor(0.0, device=emb.device)

        pos_emb = emb_seed[y]    
        neg_emb = emb_seed[~y]   

        # Fraud-fraud similarity
        pos_sim = torch.mm(pos_emb, pos_emb.t()) / self.temperature
        n       = pos_emb.size(0)
        eye     = torch.eye(n, device=emb.device).bool()
        pos_sim = pos_sim.masked_fill(eye, 1.0)
        pos_loss = (1.0 - pos_sim).clamp(min=0).mean()

        # Fraud-legit dissimilarity
        neg_sim  = torch.mm(pos_emb, neg_emb.t()) / self.temperature

        # Subsample legit for balancing
        max_neg = min(n_pos * 5, n_neg)
        if neg_emb.size(0) > max_neg:
            idx     = torch.randperm(neg_emb.size(0), device=emb.device)[:max_neg]
            neg_sim = torch.mm(pos_emb, neg_emb[idx].t()) / self.temperature
        neg_loss = torch.clamp(neg_sim - self.margin, min=0).mean()

        return pos_loss + neg_loss
    

# # class SupervisedContrastiveLoss(nn.Module):
#     """
#     """
#     def __init__(self, temperature: float = 0.07):
#         super().__init__()
#         self.temperature = temperature

#     def forward(
#             self,
#             emb:      torch.Tensor,
#             targets:  torch.Tensor,
#             batch_sz: int,
#     ) -> torch.Tensor:
#         emb_seed = F.normalize(emb[:batch_sz], dim=1)
#         y        = targets.bool()

#         if y.sum() < 2 or (~y).sum() < 2:
#             return torch.tensor(0.0, device=emb.device)

#         # similarity matrix (batch_sz, batch_sz)
#         sim = torch.mm(emb_seed, emb_seed.t()) / self.temperature

#         n   = batch_sz
#         eye = torch.eye(n, device=emb.device).bool()

#         # positive pairs: same class (fraud-fraud або legit-legit)
#         pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye

#         sim_max  = sim.detach().max(dim=1, keepdim=True).values
#         sim_exp  = torch.exp(sim - sim_max)

#         denom    = (sim_exp * ~eye).sum(dim=1, keepdim=True)
#         log_prob = sim - sim_max - torch.log(denom + 1e-8)

#         n_pos = pos_mask.sum(dim=1).float().clamp(min=1)
#         loss  = -(log_prob * pos_mask).sum(dim=1) / n_pos

#         return loss.mean()
    

### =============== HGNN MODEL ===============

class HGNN(nn.Module):
    def __init__(
            self,
            entity_cols:   list,
            in_dim:        int,
            config,
            edge_types:    list = None,
            edge_attr_dim: int  = 3,
    ):
        super().__init__()
        self.entity_cols = entity_cols
        self.config      = config
        hidden_dim       = config.hidden_dim
        dropout          = config.dropout
        heads            = config.heads
        aggr             = config.aggr
        num_layers       = config.num_layers

        assert hidden_dim % heads == 0, \
            f"hidden_dim={hidden_dim} must be divisible by heads={heads}"

        # Edge type classification
        self.temporal_edge_types = [
            et[1] for et in (edge_types or [])
            if et[1].startswith("temporal_")
        ]
        if not self.temporal_edge_types:
            self.temporal_edge_types = ["temporal_user_id", "temporal_device_id"]

        self.similarity_edge_types = [
            et[1] for et in (edge_types or [])
            if et[1].startswith("sim_")
        ]
        self.has_similarity = len(self.similarity_edge_types) > 0

        gat_out = hidden_dim // heads

        # GAT factory
        def make_gat() -> GATv2Conv:
            return GATv2Conv(
                (-1, -1),
                gat_out,
                heads          = heads,
                edge_dim       = edge_attr_dim,
                add_self_loops = False,
                concat         = True,
            )

        # Conv factories
        def make_temporal_conv() -> HeteroConv:
            return HeteroConv({
                ("transaction", et, "transaction"): make_gat()
                for et in self.temporal_edge_types
            }, aggr=aggr)

        def make_entity_conv(col: str) -> HeteroConv:
            """
            TX → entity → TX in one HeteroConv.
              Layer 1: tx aggregates from entity_old   (1-hop)
              Layer 2: tx aggregates from entity_new   (2-hop)
              Layer 3: additional context
            """
            return HeteroConv({
                ("transaction", f"to_{col}",  col): make_gat(),
                (col, f"rev_{col}", "transaction"): make_gat(),
            }, aggr=aggr)

        def make_similarity_conv() -> HeteroConv:
            return HeteroConv({
                ("transaction", et, "transaction"): make_gat()
                for et in self.similarity_edge_types
            }, aggr=aggr)

        # Temporal stream
        self.temporal_convs = nn.ModuleList([
            make_temporal_conv() for _ in range(num_layers)
        ])
        self.temporal_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.temporal_drops = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers - 1)
        ])

        # Entity streams
        self.entity_convs = nn.ModuleDict({
            col: nn.ModuleList([
                make_entity_conv(col) for _ in range(num_layers)
            ])
            for col in entity_cols
        })
        self.entity_norms = nn.ModuleDict({
            col: nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
            for col in entity_cols
        })
        self.entity_drops = nn.ModuleDict({
            col: nn.ModuleList([
                nn.Dropout(dropout) for _ in range(num_layers - 1)
            ])
            for col in entity_cols
        })

        # Similarity stream 
        if self.has_similarity:
            self.sim_convs = nn.ModuleList([
                make_similarity_conv() for _ in range(num_layers)
            ])
            self.sim_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
            self.sim_drops = nn.ModuleList([
                nn.Dropout(dropout) for _ in range(num_layers - 1)
            ])

        # Embedding dimensions
        n_streams = (
            1                                    # temporal
            + len(entity_cols)                   # entity streams
            + (1 if self.has_similarity else 0)  # similarity
        )
        concat_dim = hidden_dim * n_streams

        self.emb_dim_semantic = concat_dim   # SHAP / interpretability
        self.emb_dim_compact  = hidden_dim   # XGBoost
        self.emb_dim          = concat_dim   

        # Stream metadata 
        self.stream_names = (
            ["temporal"]
            + [f"entity_{col}" for col in entity_cols]
            + (["similarity"] if self.has_similarity else [])
        )
        self.stream_dim = hidden_dim

        # Graph projection (for XGBoost / embeddings) 
        self.graph_proj = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Classification projection (for classifier)
        self.cls_proj = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Classifier 
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _run_stream(
            self,
            convs:           nn.ModuleList,
            norms:           nn.ModuleList,
            drops:           nn.ModuleList,
            x_dict:          dict,
            edge_index_dict: dict,
            edge_attr_dict:  dict | None,
            stream_name:     str = "unknown",
    ) -> torch.Tensor:
        """
        Forward pass for one stream.
        """
        tx           = x_dict["transaction"]  # fallback
        n_successful = 0

        for i, conv in enumerate(convs):
            try:
                out = conv(x_dict, edge_index_dict, edge_attr_dict)
            except Exception as e:
                if n_successful == 0:
                    raise RuntimeError(
                        f"Stream '{stream_name}' failed on layer 0. "
                        f"Check edge_types in HeteroConv. Error: {e}"
                    )
                import warnings
                warnings.warn(
                    f"Stream '{stream_name}' stopped at layer {i} "
                    f"(after {n_successful} successful layers): {e}"
                )
                break

            if "transaction" not in out:
                if n_successful == 0:
                    raise RuntimeError(
                        f"Stream '{stream_name}' layer {i}: "
                        f"'transaction' not in conv output. "
                        f"Available: {list(out.keys())}"
                    )
                import warnings
                warnings.warn(
                    f"Stream '{stream_name}' layer {i}: "
                    f"'transaction' missing — stopping."
                )
                break

            new_tx = out["transaction"]
            new_tx = norms[i](new_tx)
            new_tx = F.gelu(new_tx)

            if i < len(drops):
                new_tx = drops[i](new_tx)

            tx     = new_tx
            x_dict = {**x_dict, "transaction": tx}
            n_successful += 1

        return tx

    def forward(
            self,
            x_dict:          dict,
            edge_index_dict: dict,
            edge_attr_dict:  dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        """
        streams = []

        # Temporal stream
        x_temporal = self._run_stream(
            self.temporal_convs,
            self.temporal_norms,
            self.temporal_drops,
            x_dict, edge_index_dict, edge_attr_dict,
            stream_name="temporal",
        )
        streams.append(x_temporal)

        # Entity streams
        for col in self.entity_cols:
            x_entity = self._run_stream(
                self.entity_convs[col],
                self.entity_norms[col],
                self.entity_drops[col],
                x_dict, edge_index_dict, edge_attr_dict,
                stream_name=f"entity_{col}",
            )
            streams.append(x_entity)

        # Similarity stream
        if self.has_similarity:
            x_sim = self._run_stream(
                self.sim_convs,
                self.sim_norms,
                self.sim_drops,
                x_dict, edge_index_dict, edge_attr_dict,
                stream_name="similarity",
            )
            streams.append(x_sim)

        # Disentangled semantic embeddings
        x_semantic = torch.cat(streams, dim=1)  # (N, concat_dim)

        # Graph projection (XGBoost / embeddings)
        x_graph = self.graph_proj(x_semantic)   # (N, hidden_dim)  


        # Classification projection 
        x_cls  = self.cls_proj(x_semantic)      # (N, hidden_dim)
        logits = self.classifier(x_cls).squeeze(-1)  # (N,)

        return logits, x_semantic, x_graph


### =============== LINK PREDICTION LOSS ===============

def link_prediction_loss(emb: torch.Tensor,
                         edge_index: torch.Tensor,
                         neg_ratio: int = 5) -> torch.Tensor:
    """
    Link prediction contrastive loss.
    """
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=emb.device)

    n   = emb.size(0)
    # Cosine similarity
    emb_n = F.normalize(emb, p=2, dim=1)
    src = edge_index[0].clamp(max=n - 1)
    dst = edge_index[1].clamp(max=n - 1)
    n_pos = src.size(0)
    n_neg = n_pos * neg_ratio

    # Positive scores
    pos_score = (emb_n[src] * emb_n[dst]).sum(dim=1)

    # Negative sampling with self-loops filter
    oversample = max(int(n_neg * 1.5), n_neg + 10)
    neg_src = torch.randint(0, n, (oversample,), device=emb.device)
    neg_dst = torch.randint(0, n, (oversample,), device=emb.device)
    no_self = neg_src != neg_dst
    neg_src = neg_src[no_self][:n_neg]
    neg_dst = neg_dst[no_self][:n_neg]

    neg_score = (emb_n[neg_src] * emb_n[neg_dst]).sum(dim=1)

    # BCE loss with logits
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score),
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_score, torch.zeros_like(neg_score),
    )

    return pos_loss + neg_loss


### =============== TRAINING LOOP ===============

def get_embedding_col_names(model: HGNN) -> tuple[list, list]:
    """
    Returns names for embeddings.
    """
    # Semantic — per stream
    semantic_cols = []
    for stream_name in model.stream_names:
        for i in range(model.stream_dim):
            semantic_cols.append(f"gnn_{stream_name}_{i}")

    # Compact
    compact_cols = [f"gnn_compact_{i}" for i in range(model.emb_dim_compact)]

    return semantic_cols, compact_cols


def build_criterion(config, y_train) -> nn.Module:     # TrainConfig
    n_pos      = int(y_train.sum())
    n_neg      = int(len(y_train) - n_pos)
    
    if config.pos_weight_override is not None:
        pos_weight = config.pos_weight_override
    elif config.use_imbalanced_sampler:
        pos_weight = 1.0
        print(f"ImbalancedSampler turns on → pos_weight=1.0")
    else:
        pos_weight = n_neg / max(n_pos, 1)
        print(f"No sampler → pos_weight={pos_weight:.1f}")

    if config.loss_type == "focal":
        return FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            pos_weight=pos_weight,
        )
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
    )


def build_optimizer(model, config):     # TrainConfig
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


def _get_edge_attr(batch) -> dict | None:
    edge_attr_dict = {
        et: batch[et].edge_attr
        for et in batch.edge_types
        if hasattr(batch[et], "edge_attr") and batch[et].edge_attr is not None
    }
    return edge_attr_dict or None


def _collect_embeddings(
        model:       nn.Module,
        data:        HeteroData,
        device,
        mask_type:   str,
        train_config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect embeddings after training.
    """
    loader = get_loader(data, mask_type=mask_type, config=train_config)
    n_nodes = data["transaction"].num_nodes
    mask    = _get_mask(data, mask_type)
    n_seeds = mask.sum().item()
    result_semantic = np.zeros((n_seeds, model.emb_dim_semantic), dtype=np.float32)
    result_graph    = np.zeros((n_seeds, model.emb_dim_compact),  dtype=np.float32)
    result_logits   = np.zeros(n_seeds,                           dtype=np.float32)

    global_to_pos                 = torch.full((n_nodes,), -1, dtype=torch.long)
    seed_positions                = mask.nonzero(as_tuple=True)[0]
    global_to_pos[seed_positions] = torch.arange(n_seeds, dtype=torch.long)

    with eval_mode(model), torch.no_grad():
        for batch in loader:
            batch          = batch.to(device)
            edge_attr_dict = _get_edge_attr(batch)

            logits, x_semantic, x_graph = model(
                batch.x_dict,
                batch.edge_index_dict,
                edge_attr_dict,
            )

            batch_sz  = batch["transaction"].batch_size
            actual_sz = min(batch_sz, logits.size(0), x_semantic.size(0))
            if actual_sz == 0:
                continue

            global_ids = batch["transaction"].n_id[:actual_sz].cpu()
            positions  = global_to_pos[global_ids]
            valid      = positions >= 0

            if not valid.all():
                print(f"{(~valid).sum().item()} invalid")

            valid_pos = positions[valid].numpy()

            result_semantic[valid_pos] = x_semantic[:actual_sz][valid].cpu().numpy()
            result_graph[valid_pos]    = x_graph[:actual_sz][valid].cpu().numpy()
            result_logits[valid_pos]   = logits[:actual_sz][valid].cpu().numpy()

    return result_semantic, result_graph, result_logits


def _compute_metrics(logits: torch.Tensor, labels: np.ndarray) -> dict:
    """Compute ROC-AUC and PR-AUC."""
    probas = torch.sigmoid(logits.detach().cpu()).numpy()
    try:
        roc = roc_auc_score(labels, probas)
    except Exception:
        roc = 0.0
    try:
        pr = average_precision_score(labels, probas)
    except Exception:
        pr = 0.0
    return {"roc_auc": roc, "pr_auc": pr}
 
 
def _is_better(current: float, best: float, monitor: str) -> bool:
    if monitor == "val_loss":
        return current < best
    return current > best   # roc_auc, pr_auc
 
 
def _best_init(monitor: str) -> float:
    if monitor == "val_loss":
        return float("inf")
    return 0.0


def train_gnn(data, y_train, device, train_config, model_config, graph_config) -> "HGNN":           
    entity_cols     = graph_config.entity_cols
    graph_features  = graph_config.graph_features
    lp_weight       = train_config.lp_weight
    monitor         = train_config.monitor_metric

    utils.set_seed()
    torch.cuda.empty_cache()

    model = HGNN(
        entity_cols   = entity_cols,
        in_dim        = len(graph_features),
        config        = model_config,
        edge_types    = list(data.edge_types),
        edge_attr_dim = graph_config.edge_attr_dim,
        ).to(device)
    
    # Loss / Optimizer
    criterion = build_criterion(train_config, y_train).to(device)
    optimizer = build_optimizer(model, train_config)

    train_loader = get_loader(data, "train", train_config, y_train)
    val_loader   = get_loader(data, "val",   train_config)
    
    contrastive_criterion = FraudContrastiveLoss(
        margin      = 0.3,    
        temperature = 0.1,
    ).to(device)
    # contrastive_criterion = SupervisedContrastiveLoss(
    #     temperature = 0.07,
    # ).to(device) 


    def run_batch(batch):
        edge_attr = _get_edge_attr(batch)
        logits, x_semantic, x_graph = model(
            batch.x_dict, batch.edge_index_dict, edge_attr
        )
        batch_sz = batch["transaction"].batch_size

        assert x_semantic.size(0) == batch["transaction"].num_nodes

        logits_seed = logits[:batch_sz]
        y           = batch["transaction"].y[:batch_sz]
        loss_cls    = criterion(logits_seed, y)

        if lp_weight > 0.0:
            loss_con = contrastive_criterion(x_graph, y, batch_sz)
            loss     = loss_cls + lp_weight * loss_con
        else:
            loss = loss_cls

        return loss, batch_sz

    def train_epoch() -> float:
        model.train()
        total_loss, total_n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, batch_sz = run_batch(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_sz
            total_n    += batch_sz
        return total_loss / max(total_n, 1)
    

    def eval_epoch() -> tuple[float, float, float]:
        """Validation loop. """
        total_loss, total_n = 0.0, 0
        all_logits, all_labels = [], []

        with eval_mode(model), torch.no_grad():
            for batch in val_loader:
                batch        = batch.to(device)
                edge_attr    = _get_edge_attr(batch)
                logits, _, _ = model(batch.x_dict, batch.edge_index_dict, edge_attr)
                batch_sz     = batch["transaction"].batch_size
                logits_seed  = logits[:batch_sz]
                y            = batch["transaction"].y[:batch_sz]
                loss         = criterion(logits_seed, y)

                total_loss += loss.item() * batch_sz
                total_n    += batch_sz

                all_logits.append(logits_seed.cpu())
                all_labels.append(y.cpu())

        avg_loss   = total_loss / max(total_n, 1)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels).numpy()
        metrics    = _compute_metrics(all_logits, all_labels)

        return avg_loss, metrics["roc_auc"], metrics["pr_auc"]


    def get_monitor_value(val_loss, roc_auc, pr_auc) -> float:
        return {"val_loss": val_loss, "roc_auc": roc_auc, "pr_auc": pr_auc}[monitor]
    

    # Phase 1 — Cosine Warm Restarts 
    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_config.t0_cosine,       
        T_mult=train_config.t_mult,    
        eta_min=1e-5  
    )

    header = f"{'Ep':>5} | {'Train':>8} | {'ValLoss':>8} | {'ROC-AUC':>8} | {'PR-AUC':>7} | {'LR':>9}"
    sep    = "─" * 65
    print(f"  PHASE 1 — CosineAnnealingWarmRestarts  ({train_config.epochs_phase1} epochs)")
    print(sep)
    print(header)
    print(sep)

    best_metric = _best_init(monitor)
    best_state    = None
    no_improve = 0

    for epoch in range(train_config.epochs_phase1):
        train_loss   = train_epoch()
        val_loss, roc_auc, pr_auc = eval_epoch()
        scheduler_cos.step()
        lr_now   = optimizer.param_groups[0]["lr"]
        monitor_val     = get_monitor_value(val_loss, roc_auc, pr_auc)

        marker = ""
        if _is_better(monitor_val, best_metric, monitor):
            best_metric = monitor_val
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker        = " ✅" 
            no_improve   = 0
        else:
            no_improve  += 1
        print(f"{epoch:6d} | {train_loss:8.4f} | {val_loss:8.4f} | "
              f"{roc_auc:8.4f} | {pr_auc:7.4f} | {lr_now:9.6f}{marker}")

        if no_improve >= train_config.early_stopping_patience:
            print(f"\n  ⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {train_config.early_stopping_patience} epochs)")
            break
        
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n Phase 1 best restored ({monitor}={best_metric:.4f})")


    # PHASE 2 — ReduceLROnPlateau (fine-tuning)
    for pg in optimizer.param_groups:
        pg["lr"] = train_config.lr * 0.1  

    plateau_mode = "min" if monitor == "val_loss" else "max"
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = plateau_mode,
        factor   = train_config.plateau_factor,      # 0.7,
        patience = train_config.plateau_patience,  #  8,
        min_lr   = 1e-6,
        # verbose=True
    )

    print(f"\n{'═'*55}")
    print(f"  PHASE 2 — ReduceLROnPlateau  ({train_config.epochs_phase2} epochs)")
    print(sep)
    print(header)
    print(sep)

    no_improve = 0
    best_p2 = best_metric   

    for epoch in range(train_config.epochs_phase2):
        train_loss                = train_epoch()
        val_loss, roc_auc, pr_auc = eval_epoch()
        monitor_val               = get_monitor_value(val_loss, roc_auc, pr_auc)
        scheduler_plateau.step(monitor_val)   
        lr_now = optimizer.param_groups[0]["lr"]

        marker = ""
        if _is_better(monitor_val, best_p2, monitor):
            best_p2    = monitor_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker       = " ✅"
            no_improve   = 0
        else:
            no_improve  += 1

        print(f"{epoch:5d} | {train_loss:8.4f} | {val_loss:8.4f} | "
              f"{roc_auc:8.4f} | {pr_auc:7.4f} | {lr_now:9.6f}{marker}")
 
        if no_improve >= train_config.early_stopping_patience:
            print(f"\n  ⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {train_config.early_stopping_patience} epochs)")
            break

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n Best model restored ({monitor}={best_p2:.4f})")

    train_semantic, train_graph, train_logits = _collect_embeddings(
    model, data, device, "train", train_config
    )
    val_semantic, val_graph, val_logits = _collect_embeddings(
        model, data, device, "val", train_config
    )

    return model, {
        "train_emb":      train_semantic,  # SHAP
        "val_emb":        val_semantic,
        "train_emb_graph": train_graph,   # XGBoost
        "val_emb_graph":   val_graph,
        "train_logits":   train_logits,
        "val_logits":     val_logits,
    }


def validate_fraud_ratio(data, train_config, y_train, expected_baseline: float = 0.035):
    loader = get_loader(data, "train", train_config, y_train)

    fraud_ratios = []
    for i, batch in enumerate(loader):
        if i >= 20:  
            break

        batch_sz = batch["transaction"].batch_size
        targets  = batch["transaction"].y[:batch_sz]
        ratio    = (targets == 1).float().mean().item()
        fraud_ratios.append(ratio)

    mean_ratio = np.mean(fraud_ratios)
    # uplift     = mean_ratio / expected_baseline

    # print(f"  Baseline fraud ratio : {expected_baseline:.3f}")
    # print(f"  Sampled fraud ratio  : {mean_ratio:.3f}")
    # print(f"  Uplift               : {uplift:.1f}x")

    # if uplift > 1.5:
    #     print(" ImbalancedSampler works")
    # else:
    #     print("Uplift too low")

    return mean_ratio


def validate_graph_and_loader(
        data:              HeteroData, 
        train_config, 
        y_train                = None,
        expected_edge_dim: int = 3,) -> None:
    """
    """
    print("\n🔍 Validating graph and loaders...")
 
    assert data["transaction"].x        is not None, "No transaction features!"
    assert data["transaction"].y        is not None, "No transaction labels!"
    assert data["transaction"].train_mask is not None, "No train mask!"
    assert data["transaction"].val_mask   is not None, "No val mask!"
 
    n_train = data["transaction"].train_mask.sum().item()
    n_val   = data["transaction"].val_mask.sum().item()
    n_total = data["transaction"].num_nodes
    print(f"Nodes: {n_total:,}  train={n_train:,}  val={n_val:,}")
    print(f"Transaction x: {data['transaction'].x.shape}")
    print(f"Fraud ratio  : {data['transaction'].y.mean():.4f}")
 
    node_sizes = {nt: data[nt].num_nodes for nt in data.node_types}
 
    for et in data.edge_types:
        src_type, _, dst_type = et
        ei = data[et].edge_index

        assert ei.size(0) == 2, f"{et}: wrong edge_index shape"

        if ei.size(1) == 0:
            print(f"{str(et)[-35:]:>35}: empty")
            continue

        assert ei.min() >= 0, f"{et}: negative edge index"
        assert ei[0].max() < node_sizes[src_type], (
            f"{et}: src index {ei[0].max().item()} >= "
            f"num_{src_type}_nodes={node_sizes[src_type]}"
        )
        assert ei[1].max() < node_sizes[dst_type], (
            f"{et}: dst index {ei[1].max().item()} >= "
            f"num_{dst_type}_nodes={node_sizes[dst_type]}"
        )

        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            ea = data[et].edge_attr
            assert ea.size(0) == ei.size(1), \
                f"{et}: edge_attr rows {ea.size(0)} != edges {ei.size(1)}"
            assert ea.size(1) == expected_edge_dim, \
                f"{et}: edge_attr dim {ea.size(1)} != {expected_edge_dim}"
            assert not torch.isnan(ea).any(), \
                f"{et}: NaN in edge_attr"
            assert not torch.isinf(ea).any(), \
                f"{et}: Inf in edge_attr"

        print(f"{str(et)[-35:]:>35}: {ei.size(1):,} edges")
    
    train_loader = get_loader(data, "train", train_config, y_train)
    val_loader   = get_loader(data, "val",   config=train_config)
 
    train_batch = next(iter(train_loader), None)
    val_batch   = next(iter(val_loader),   None)
 
    assert train_batch is not None, "Empty train loader!"
    assert val_batch   is not None, "Empty val loader!"
    assert train_batch["transaction"].batch_size > 0
    assert len(train_batch.edge_index_dict) > 0
 
    print(f"Train batch: {train_batch['transaction'].batch_size} seed nodes")
    print(f"Val batch  : {val_batch['transaction'].batch_size} seed nodes")
 
    for et in train_batch.edge_types:
        es = train_batch[et]
        if hasattr(es, "edge_attr") and es.edge_attr is not None:
            print(f"edge_attr {str(et)[-25:]:>25}: {es.edge_attr.shape}")

    # Check ImbalancedSampler
    if train_config.use_imbalanced_sampler and y_train is not None:
        y_np    = np.array(y_train)
        n_fraud = y_np.sum()
        n_total_train = len(y_np)
        baseline_ratio = n_fraud / n_total_train

        fraud_in_batch = train_batch["transaction"].y[
            :train_batch["transaction"].batch_size
        ].mean().item()

        uplift = fraud_in_batch / (baseline_ratio + 1e-8)
        print(f"\n  Baseline fraud ratio : {baseline_ratio:.3f}")
        print(f"  Sampled fraud ratio  : {fraud_in_batch:.3f}")
        print(f"  Uplift               : {uplift:.1f}x")
        if uplift > 2.0:
            print(f"ImbalancedSampler works")
        else:
            print(f"Uplift too low")


### =============== EXTRACT EMBEDDINGS + LOGITS ===============



### =============== GRAPH VISUALISATION ===============

def save_graph_drawing(g, location):
    plt.figure(figsize=(12, 8))
    node_colors = {node: 0.0 if 'user' in node else 0.5 for node in g.nodes()}
    nx.draw(g, node_size=10000, pos=nx.spring_layout(g), with_labels=True, font_size=14,
            node_color=list(node_colors.values()), font_color='white')
    plt.savefig(location, bbox_inches='tight')
