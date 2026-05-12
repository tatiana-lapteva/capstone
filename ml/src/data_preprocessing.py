import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass, field


## =============== CONSTANTS ===============
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY  = 86400
HOURS_PER_WEEK   = 168


@dataclass
class PreprocessConfig:
    second_per_hour  = 3600
    seconds_per_day  = 86400
    hours_per_week   = 168
    cat_cols: list = field(default_factory=list)
    card_cols: list = field(default_factory=list)
    node_source_cols: list = field(default_factory=list)

    target_enc_cols: list = field(default_factory=list)
    target_enc_num_cols: list = field(default_factory=list)
    label_enc_cols: list = field(default_factory=list)
    m_cols_binary: list = field(default_factory=list)
    binary_id_cols: list = field(default_factory=list)


@dataclass
class PreprocessArtifacts:
    num_fill: dict
    cat_fill: dict
    encoders: dict
    encoded_cols: list
    log_artifacts: dict
    card_stability_map: object
    feature_cols: list
    cat_cols: list


## =============== DATA LOADING ===============
def load_data(path='../data/raw/IEEE-CIS Fraud Detection/'):
    train_trans = pd.read_csv(f'{path}train_transaction.csv')
    train_ident = pd.read_csv(f'{path}train_identity.csv')
    test_trans  = pd.read_csv(f'{path}test_transaction.csv')
    test_ident  = pd.read_csv(f'{path}test_identity.csv')
    
    # Fix columns names in Test Dataset
    test_ident.columns = test_ident.columns.str.replace("-", "_")

    train = train_trans.merge(train_ident, on='TransactionID', how='left')
    test  = test_trans.merge(test_ident,  on='TransactionID', how='left')
    
    return train, test


## =============== DATATIME TRANSFORMATION - D-COLUMNS FIX ===============
def d_columns_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms D columns from relative time deltas to absolute dates.
    Eliminates train/test shift caused by different reference points.
    Must be called BEFORE fill_numeric_features.
    """
    df = df.copy()
    d_cols = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
    for col in d_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DT_day"] = (df["TransactionDT"] / SECONDS_PER_DAY).astype("float32")  

    for col in d_cols:
        df[col] = (df["DT_day"] - df[col]).astype("float32")
    if "D1" in df.columns:
        df["D1"] = df["D1"].astype("float32")
        df["D1"] = df["D1"].round(0)
    return df


## =============== DROP LEAKY RAW FEATURES ===============
def drop_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that let the model distinguish train from test."""
    leaky_features = ['V6', 'V8', 'V9']     
    
    return df.drop(columns=[c for c in leaky_features if c in df.columns])


## =============== CREATE COMPOSITE ID FEATURE ===============
def create_composite_id(
        df:      pd.DataFrame,
        cols:    list,
        id_name: str,
        ) -> pd.DataFrame:
    """Proxy for a unique cardholder"""
    missing = [c for c in  cols if c not in df.columns]
    assert not missing, f"Missing Columns: {missing}"
    df = df.copy()
    result = df[cols[0]].astype(str).replace("nan", "unknown")
    for col in cols[1:]:
        result = result + "_" + df[col].astype(str).replace("nan", "unknown")
    df[id_name] = result
    return df


def compute_fraud_rate_oof(
        df:       pd.DataFrame,
        col:      str,
        n_splits: int = 5,
) -> np.ndarray:
    """
    Out-of-fold fraud rate .
    TimeSeriesSplit for temporal order
    """
    oof         = np.zeros(len(df))
    global_mean = float(df["isFraud"].mean())
    tscv        = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, val_idx in tscv.split(df):
        fold_map     = df.iloc[train_idx].groupby(col)["isFraud"].mean()
        fold_mean    = float(df.iloc[train_idx]["isFraud"].mean())  # fold mean
        oof[val_idx] = (
            df.iloc[val_idx][col]
            .map(fold_map)
            .fillna(fold_mean)   
            .values
        )

    n_first = len(df) - tscv.split(df).__next__()[1].size * n_splits
    uncovered = (oof == 0.0)
    oof[uncovered] = global_mean

    return oof


def create_card_features(
        df:        pd.DataFrame,
        card_cols: list,
) -> pd.DataFrame:
    """
    Generate card-based features.
    """
    df = df.copy()

    for col in card_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .replace("nan",  "unknown")
                .replace("<NA>", "unknown")
            )

    if "card1" in df.columns and "card2" in df.columns:
        df["card1_card2"] = (
            df["card1"].astype(str) + "_" + df["card2"].astype(str)
        )

    if "card1" in df.columns and "addr1" in df.columns and "isFraud" in df.columns:
        card_stability = df.groupby("card1")["addr1"].nunique()
        df["card_stability"] = (
            df["card1"].map(card_stability).fillna(1).astype("float32")
        )
    else:
        df["card_stability"] = 1.0

    return df


## =============== CREATE TIME FEATURES ===============
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclic / calendar features derived purely from TransactionDT.
    Works on a DataFrame; call AFTER split so indices remain aligned.
    """
    df = df.copy()
    df["day"]         = (df["TransactionDT"] // SECONDS_PER_DAY).astype("int32")
    df["is_weekend"] = (((df["TransactionDT"] // SECONDS_PER_DAY) % 7) >= 5).astype("int8")
    df["day_of_week"] = ((df["TransactionDT"] // SECONDS_PER_DAY) % 7).astype("int8")
    hour_frac       = (df["TransactionDT"] % SECONDS_PER_DAY) / SECONDS_PER_DAY

    df['hour_sin'] = np.sin(2 * np.pi * hour_frac).astype("float32")
    df['hour_cos'] = np.cos(2 * np.pi * hour_frac).astype("float32")
    df['week_sin'] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    df['week_cos'] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float32")

    # df["time_bucket"] = (df["TransactionDT"] // SECONDS_PER_HOUR % HOURS_PER_WEEK).astype("int16")
    
    # df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    
    # df["is_night"]   = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype("int8")

    return df


def compute_tx_count_1h(df: pd.DataFrame, window: int = SECONDS_PER_HOUR) -> pd.DataFrame:
    """
    """
    df       = df.sort_values(["user_id", "TransactionDT"]).copy()
    tx_count = np.zeros(len(df), dtype=np.int32)

    for _, idx in df.groupby("user_id").groups.items():
        times  = df.loc[idx, "TransactionDT"].values
        left, counts = 0, []
        for right in range(len(times)):
            while times[right] - times[left] > window:
                left += 1
            counts.append(right - left + 1)
        tx_count[idx] = counts

    df["tx_count_1h"] = tx_count
    return df


def velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Velocity Features.
    """
    df = df.copy()
    df["log_tx_count"] = np.log1p(df["tx_count_1h"].fillna(0)).astype("float32")
    df["tx_velocity"] = df["tx_count_1h"] / (df["dt_prev"].clip(1, 1e6) / SECONDS_PER_HOUR).astype("float32")  # tx/hour
    return df


def compute_window_features(
        df:     pd.DataFrame,
        window: int = 86400,   # 24h
) -> pd.DataFrame:
    """
    Rolling window features for 24 hours by user_id.
    """
    df = df.sort_values(["user_id", "TransactionDT"]).copy()

    tx_count_24h = np.zeros(len(df), dtype=np.int32)
    amt_sum_24h  = np.zeros(len(df), dtype=np.float32)
    amt_mean_24h = np.zeros(len(df), dtype=np.float32)

    for _, idx in df.groupby("user_id").groups.items():
        times  = df.loc[idx, "TransactionDT"].values
        amts   = df.loc[idx, "TransactionAmt"].values
        left   = 0

        for right in range(len(times)):
            while times[right] - times[left] > window:
                left += 1
            window_amts = amts[left:right]

            tx_count_24h[idx[right]] = right - left
            amt_sum_24h[idx[right]]  = float(window_amts.sum())  if len(window_amts) > 0 else 0.0
            amt_mean_24h[idx[right]] = float(window_amts.mean()) if len(window_amts) > 0 else 0.0

    df["tx_count_24h"] = tx_count_24h
    df["amt_sum_24h"]  = amt_sum_24h.astype("float32")
    df["amt_mean_24h"] = amt_mean_24h.astype("float32")

    return df


def compute_amt_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score of transaction amount relative to rolling 24h mean/std.
    Feature amt_mean_24h is required.
    """
    df = df.copy()
    if "amt_rolling_std" in df.columns and "amt_mean_24h" in df.columns:
        std = df["amt_rolling_std"].replace(-1, np.nan)  # -1 = немає історії
        mean = df["amt_mean_24h"]

        df["amt_zscore"] = (
            (df["TransactionAmt"] - mean)
            / (std + 1e-8)
        ).clip(-10, 10).fillna(0).astype("float32")
    else:
        df["amt_zscore"] = 0.0

    return df


## =============== CREATE ROLLING FEATURES ===============
def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["orig_idx"] = np.arange(len(df))

    # Card_ID
    cat_card_cols = ["card1", "card2", "card3", "card5", "card4", "card6"]
    for col in cat_card_cols:
        if col not in df.columns:
            continue
        df[col] = (
            df[col].astype(str).fillna("Unknown")
        )
    cat_part = df[cat_card_cols].astype(str)
    df["Card_ID"] = cat_part.agg("_".join, axis=1)
    df = df.sort_values(["Card_ID", "TransactionDT"])
  
    rolling_mean = df.groupby("Card_ID")["TransactionAmt"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    rolling_std = df.groupby("Card_ID")["TransactionAmt"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).std()
    )
    rolling_min = df.groupby("Card_ID")["TransactionAmt"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).min()
    )
    rolling_max = df.groupby("Card_ID")["TransactionAmt"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).max()
    )

    df["amt_diff_mean"] = df["TransactionAmt"] - rolling_mean
    df["amt_ratio_mean"] = df["TransactionAmt"] / (rolling_mean + 1)        # +1 avoids huge values 
    df["amt_rolling_std"] = rolling_std
    df["amt_rolling_min"] = rolling_min
    df["amt_rolling_max"] = rolling_max
    df["no_history"] = (df.groupby("Card_ID")["TransactionAmt"]
                        .transform("cumcount") == 0).astype(int)

    df["amt_diff_mean"]   = df['amt_diff_mean'].fillna(1)
    df["amt_ratio_mean"]  = df["amt_ratio_mean"].fillna(1)
    df["amt_rolling_std"] = df["amt_rolling_std"].fillna(-1)
    df["amt_rolling_min"] = df["amt_rolling_min"].fillna(df["TransactionAmt"])
    df["amt_rolling_max"] = df["amt_rolling_max"].fillna(df["TransactionAmt"])

    # Return original indexes
    df = df.sort_values("orig_idx")
    df = df.drop(columns=["orig_idx", "Card_ID"])
    return df


## =============== TRAIN/VAL SPLIT DATASET ===============
def temporal_split(df: pd.DataFrame) -> tuple:
    """ Temporal split by TransactionDT. """
    split_ratio: float = 0.8
    df = df.copy()
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    threshold = df['TransactionDT'].quantile(split_ratio)
    train = df[df['TransactionDT'] <= threshold].reset_index(drop=True)
    val   = df[df['TransactionDT'] >  threshold].reset_index(drop=True)
    
    return train, val


## =============== FILL MISSING NUMERIC FEATURES: TRAIN DATASET ===============
def fill_numeric_features(
        df:         pd.DataFrame,
        num_cols:   list,
        strategy:   str = "constant",
        drop_ratio: float = 0.98,
        ) -> tuple[pd.DataFrame, dict]:
    """
    Fit imputation on train; returns filled df + fill_values dict.
    Also adds <col>_is_missing binary indicators.
    """
    df = df.copy()
    fill_values: dict = {}
    cols_to_drop = []
    exclude = {"isFraud", "TransactionID", "TransactionDT"}
    n_cols = [
        c for c in num_cols
            if c in df.columns and c not in exclude
    ]
    for col in n_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        missing_ratio = df[col].isna().mean()

        # drop дуже пусті фічі
        if missing_ratio > drop_ratio:
            cols_to_drop.append(col)
            continue

        df[col + "_is_missing"] = df[col].isna().astype("int8")
        if strategy == "median":                
            value = float(df[col].median())
        elif strategy == "mean":                
            value = float(df[col].mean())
        elif strategy == "constant":
            col_min = df[col].min()
            value: float = float(col_min - 1) if pd.notna(col_min) else -1.0       
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")
        df[col] = df[col].fillna(value)
        fill_values[col] = value
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        # print(f"Numeric dropped ({len(cols_to_drop)}): {cols_to_drop}")
    fill_values["__dropped_num_cols__"] = cols_to_drop
    return df, fill_values


## =============== FILL MISSING CATEGORICAL FEATURES: TRAIN DATASET ===============
def fill_categorical_features(
        df:         pd.DataFrame,
        cat_cols:   list,
        drop_ratio: float = 0.99,
        ) -> tuple[pd.DataFrame, dict]:
    """Fit categorical imputation on train.         strategy: str = "unknown","""
    df = df.copy()
    fill_values: dict = {}
    cols_to_drop = []  
    # m_cols = {f'M{i}' for i in range(1, 10)}
    c_cols = [c for c in cat_cols if c in df.columns]
    for col in c_cols:
        missing_ratio = (df[col].isna() | (df[col].astype(str) == "nan")).mean()
        if missing_ratio > drop_ratio:
            cols_to_drop.append(col)
            continue
        df[col] = df[col].astype(str).replace("nan", "unknown").replace("<NA>", "unknown").fillna("unknown")
        fill_values[col] = "unknown"
          
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        # print(f"Categorical dropped ({len(cols_to_drop)}): {cols_to_drop}")
    fill_values["__dropped_cat_cols__"] = cols_to_drop

    return df, fill_values


## =============== Fill Missing Values: TEST dataset  ===============
def apply_fill_values(df: pd.DataFrame,
                      num_fill_values: dict,
                      cat_fill_values: dict = None) -> pd.DataFrame:
    """Apply train-fitted fill values to val / test."""
    df = df.copy()
    dropped_num = num_fill_values.get("__dropped_num_cols__", [])
    if dropped_num:
        df = df.drop(columns=[c for c in dropped_num if c in df.columns])
    
    dropped_cat = cat_fill_values.get("__dropped_cat_cols__", [])
    if dropped_cat:
        df = df.drop(columns=[c for c in dropped_cat if c in df.columns])

    for col, value in num_fill_values.items():
        if col.startswith("__"):
            continue
        if col not in df.columns:
            continue
        df[col + "_is_missing"] = df[col].isna().astype("int8")
        df[col] = df[col].fillna(value)
    
    for col, value in cat_fill_values.items():
        if col.startswith("__"):
            continue
        if col not in df.columns:
            continue
        df[col] = df[col].fillna(value).astype(str).replace("nan", value)
    return df


def create_transaction_level_features(
        full_df: pd.DataFrame,
        n_train: int,
) -> pd.DataFrame:
    """
    Transaction-level features corresponding to entity-level features.
    """
    df = full_df.copy()
    dt = df["TransactionDT"]

    # Timespan Per User (days)
    train_mask = df.index < n_train

    user_time_span = (
        df[train_mask]
        .groupby("user_id")["TransactionDT"]
        .agg(lambda x: (x.max() - x.min()) / 86400.0 + 1.0)
    )
    user_tx_count = df[train_mask].groupby("user_id").size()
    user_tx_per_day = user_tx_count / user_time_span

    df["tx_per_day"] = (
        df["user_id"]
        .map(user_tx_per_day)
        .fillna(1.0)
        .astype("float32")
    )

    # Night transactions ratio 
    df_tmp = df.copy()
    df_tmp["hour"]     = (df_tmp["TransactionDT"] % 86400) // 3600
    df_tmp["is_night"] = df_tmp["hour"].between(0, 5).astype(int)

    user_night_ratio = (
        df_tmp[train_mask]
        .groupby("user_id")["is_night"]
        .mean()
    )

    df["night_tx_ratio"] = (
        df["user_id"]
        .map(user_night_ratio)
        .fillna(0.0)
        .astype("float32")
    )

    # Hour of Week + Cyclic

    df["hour_of_week"] = ((dt // 3600) % 168).astype("int16")
    df["how_sin"] = np.sin(
        2 * np.pi * df["hour_of_week"] / 168
    ).astype("float32")
    df["how_cos"] = np.cos(
        2 * np.pi * df["hour_of_week"] / 168
    ).astype("float32")

    # Global transaction per hour 
    hour_bucket    = dt // 3600
    train_bucket   = hour_bucket.iloc[:n_train]
    train_rate_map = train_bucket.value_counts()

    df["global_tx_per_hour"] = (
        hour_bucket
        .map(train_rate_map)
        .fillna(train_rate_map.median())  
        .astype("int32")
    )

    return df




## =============== Encoding Categorical Features  ===============


def _oof_target_encode(df, col, smoothing=10.0, n_splits=5):
    target = "isFraud"
    df_sorted  = df.sort_values("TransactionDT")
    orig_index = df_sorted.index
    df_sorted  = df_sorted.reset_index(drop=True).copy()

    # ✅ конвертуємо category → str один раз на початку
    if pd.api.types.is_categorical_dtype(df_sorted[col]):
        df_sorted[col] = df_sorted[col].astype(str)

    global_mean = float(df_sorted[target].mean())
    oof         = np.full(len(df_sorted), global_mean, dtype=np.float32)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for tr_idx, vl_idx in tscv.split(df_sorted):
        fold_train       = df_sorted.iloc[tr_idx]
        fold_global_mean = float(fold_train[target].mean())

        stats  = fold_train.groupby(col, observed=True)[target].agg(["sum", "count"])
        te_map = (
            (stats["sum"] + smoothing * fold_global_mean)
            / (stats["count"] + smoothing)
        )
        oof[vl_idx] = (
            df_sorted.iloc[vl_idx][col]
            .map(te_map)
            .fillna(fold_global_mean)
            .astype("float32")
            .values
        )

    # ── Final map — на всьому df_sorted
    stats     = df_sorted.groupby(col, observed=True)[target].agg(["sum", "count"])
    final_map = (
        (stats["sum"] + smoothing * global_mean)
        / (stats["count"] + smoothing)
    )

    return pd.Series(oof, index=orig_index), {
        "final_map":   final_map,
        "global_mean": global_mean,
    }


def categorical_encoding_train(
        df:                  pd.DataFrame,
        target_enc_cols:     list,
        target_enc_num_cols: list,
        label_enc_num_cols:  list,
        m_cols_binary:       list,
        binary_id_cols:      list,
        node_source_cols:    list,
        ) -> tuple[pd.DataFrame, dict]:
    """
    Fits and applies categorical encodings on train set.
    Returns encoded df and encoders dict for applying to val/test.
    """
    df = df.copy()
    encoders: dict = {}
    ## Target encoding — high cardinality categorical (>100)
    
    for col in target_enc_cols:
        if col not in df.columns:
            continue
        encoded, enc_info = _oof_target_encode(df, col)         # TargetEncoder(smoothing=10)
        df = df.drop(columns=[col])
        df[col] = encoded.values.astype("float32")  # .values знімає індекс і dtype
        encoders[col] = {"type": "target", **enc_info}

    # ── Target encoding: high cardinality numerical columns 
    for col in target_enc_num_cols:
        if col not in df.columns:
            continue
        encoded, enc_info = _oof_target_encode(df, col)
        df = df.drop(columns=[col])
        df[col] = encoded.values.astype("float32")
        encoders[col] = {"type": "target_num", **enc_info}

    ## Label encoding: low-medium cardinality (3-100) 
    for col in label_enc_num_cols:
        if col not in df.columns:
            continue
        enc = LabelEncoder()
        col_values = df[col].astype(str)
        enc.fit(col_values)
        df = df.drop(columns=[col])
        df[col] = enc.transform(col_values).astype("int16")
        encoders[col] = {"type": "label", "enc": enc}
  

    # ── Binary mapping: M columns (T/F) ───────────────────────
    m_mapping = {'T': 1, 'F': 0, 'unknown': -1}
    m4_mapping = {'M0': 0, 'M1': 1, 'M2': 2, 'unknown': -1}
    for col in m_cols_binary:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).map(m_mapping).fillna(-1).astype("int8")
        encoders[col] = {"type": "binary_m", "mapping": m_mapping}
    
    if "M4" in df.columns:
        df["M4"] = df["M4"].astype(str).map(m4_mapping).fillna(-1).astype("int8")
        encoders["M4"] = {"type": "binary_m4", "mapping": m4_mapping}

    # ── Binary mapping: id binary columns ────────────────────
    for col in binary_id_cols:
        if col not in df.columns:
            continue
        vals = df[col].fillna("unknown").astype(str).unique()
        mapping = {v: i for i, v in enumerate(vals)}
        df[col] = (df[col]
                   .fillna("unknown").astype(str)
                   .map(mapping)
                   .fillna(-1).astype("int8"))
        encoders[col] = {"type": "binary_id", "mapping": mapping}

    # ── Frequency encoding: card* + node source cols ─────────────────────────────
    for col in node_source_cols:
        if col not in df.columns:
            continue
        freq = df[col].astype(str).value_counts()
        df[f"{col}_freq"] = df[col].astype(str).map(freq).fillna(0).astype("int32")
        encoders[f"{col}_freq"] = {"type": "freq", "freq": freq}

    # ── Fraud rate: node source cols ──────────────────────────
    for col in node_source_cols:
        if col not in df.columns or "isFraud" not in df.columns:
            continue
        oof         = compute_fraud_rate_oof(df, col)
        global_mean = float(df["isFraud"].mean())
        smoothing   = 10.0
        stats       = df.groupby(col)["isFraud"].agg(["sum", "count"])
        fraud_map   = (
            (stats["sum"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        df[f"{col}_fraud_rate"] = oof.astype("float32")
        encoders[f"{col}_fraud_rate"] = {
            "type":        "fraud_rate",
            "fraud_map":   fraud_map,
            "global_mean": global_mean,
        }

    # ── Перевірка object dtype ────────────────────────────────
    problem_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c not in node_source_cols
    ]
    if problem_cols:
        print(f"  ⚠️  object dtype : {problem_cols}")

    encoded_cols = [
        col for col in encoders
        if encoders[col].get("type") not in ("freq", "fraud_rate")
        and col in df.columns
    ] + [
    # ✅ fraud_rate — виключити з log transform
    f"{col}_fraud_rate"
    for col in node_source_cols
    if f"{col}_fraud_rate" in df.columns
    ]

    return df, encoders, encoded_cols        


def categorical_encoding_valtest(
    df:       pd.DataFrame,
    encoders: dict,
) -> pd.DataFrame:
    """
    Apply train-fitted encoders to val/test set.
    Unseen values → -1 / 0.
    """
    df = df.copy()

    # Target encoding: categorical
    for col, enc_info in encoders.items():
        enc_type = enc_info.get("type")

        # ── freq feature — окрема назва колонки ───────────────
        if enc_type == "freq":
            # col = "card1_freq", "DeviceInfo_freq" etc.
            src_col = col.replace("_freq", "")
            if src_col in df.columns:
                df[col] = (
                    df[src_col].astype(str)
                    .map(enc_info["freq"])
                    .fillna(0).astype("int32")
                )
            continue

        # ── fraud_rate feature ────────────────────────────────
        if enc_type == "fraud_rate":
            src_col = col.replace("_fraud_rate", "")
            if src_col in df.columns:
                df[col] = (
                    df[src_col].astype(str)
                    .map(enc_info["fraud_map"])
                    .fillna(enc_info["global_mean"])
                    .astype("float32")
                )
            continue

        # ── решта — колонка з тим самим ім'ям ─────────────────
        if col not in df.columns:
            continue

        if enc_type in ("target", "target_num"):
            df[col] = (
                df[col].map(enc_info["final_map"])
                .fillna(enc_info["global_mean"])
                .astype("float32")
            )

        elif enc_type == "label":
            enc   = enc_info["enc"]
            known = set(enc.classes_)
            s     = df[col].astype(str)
            mask  = s.isin(known)
            result = pd.Series(-1, index=df.index, dtype="int16")
            result[mask] = enc.transform(s[mask]).astype("int16")
            df[col] = result

        elif enc_type == "binary_m":
            df[col] = (
                df[col].astype(str)
                .map(enc_info["mapping"])
                .fillna(-1).astype("int8")
            )

        elif enc_type == "binary_m4":
            df[col] = (
                df[col].astype(str)
                .map(enc_info["mapping"])
                .fillna(-1).astype("int8")
            )

        elif enc_type == "binary_id":
            df[col] = (
                df[col].fillna("unknown").astype(str)
                .map(enc_info["mapping"])
                .fillna(-1).astype("int8")
            )

    return df
   

def apply_log_transform(
    df:            pd.DataFrame,
    right_log_cols: list = None,   # ✅ колонки БЕЗ від'ємних → log1p
    left_log_cols:  list = None,   # ✅ колонки З від'ємними → signed log1p
    clip_quantile:  float = 0.999,
    clip_values:    dict  = None,  # ✅ для val/test — збережені квантилі
) -> tuple[pd.DataFrame, dict]:
    """
    Log трансформація skewed features без leakage.

    right_log_cols — колонки без від'ємних значень → log1p
    left_log_cols  — колонки з від'ємними значеннями → signed log1p

    Train: fit clip_quantile → зберігає в artifacts
    Val/Test: тільки transform з збережених clip_values
    """
    df        = df.copy()
    artifacts = {
        "right_log_cols": right_log_cols or [],
        "left_log_cols":  left_log_cols  or [],
        "clip_values":    dict(clip_values) if clip_values is not None else {},
    }

    # ── RIGHT → log1p ─────────────────────────────────────────
    for col in (right_log_cols or []):
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # ✅ fit тільки на train — val/test передають clip_values
        if col not in artifacts["clip_values"]:
            q_upper = float(df[col].quantile(clip_quantile))
            artifacts["clip_values"][col] = (None, q_upper)
        else:
            _, q_upper = artifacts["clip_values"][col]

        df[col] = np.log1p(df[col].clip(lower=0, upper=q_upper))

        # ✅ перевірка після трансформації
        if np.isinf(df[col]).any():
            print(f"  ⚠️  {col}: inf після log1p — замінюємо на 0")
            df[col] = df[col].replace([np.inf, -np.inf], 0)

    # ── LEFT / MIXED → signed log1p ───────────────────────────
    for col in (left_log_cols or []):
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # ✅ clip з обох боків для колонок з від'ємними значеннями
        if col not in artifacts["clip_values"]:
            q_lower = float(df[col].quantile(1 - clip_quantile))
            q_upper = float(df[col].quantile(clip_quantile))
            artifacts["clip_values"][col] = (q_lower, q_upper)
        else:
            q_lower, q_upper = artifacts["clip_values"][col]

        series  = df[col].clip(lower=q_lower, upper=q_upper)
        df[col] = np.sign(series) * np.log1p(np.abs(series))

        # ✅ перевірка після трансформації
        if np.isinf(df[col]).any():
            print(f"  ⚠️  {col}: inf після signed log1p — замінюємо на 0")
            df[col] = df[col].replace([np.inf, -np.inf], 0)

    # print(f"✓ Transformed: {len(right_log_cols or [])} right (log1p)  "
    #       f"+ {len(left_log_cols or [])} left/mixed (signed log1p)")

    return df, artifacts


def create_amount_ratio_features(
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ratio поточної суми до середньої по user/card.
    ✅ Fit тільки на train — no leakage.
    ✅ Викликати ДО log_transform.
    """
    train_df = train_df.copy()
    val_df   = val_df.copy()

    maps = {
        "user_id": train_df.groupby("user_id")["TransactionAmt"].mean(),
        "card1":   train_df.groupby("card1")["TransactionAmt"].mean(),
    }

    col_names = {
        "user_id": "amt_vs_user_mean",
        "card1":   "amt_vs_card_mean",
    }

    for group_col, mean_map in maps.items():
        col_name = col_names[group_col]

        for df in [train_df, val_df]:
            denom = df[group_col].map(mean_map).fillna(df["TransactionAmt"])
            df[col_name] = (
                df["TransactionAmt"] / (denom + 1e-8)
            ).clip(0, 100).astype("float32")

    return train_df, val_df


def analyze_skewness(
    df:              pd.DataFrame,
    exclude_cols:    list  = None,
    skew_threshold:  float = 0.5,
    log_threshold:   float = 2.0,
) -> tuple[pd.DataFrame, list, list]:
    """
    Аналізує skewness числових колонок датасету.

    Parameters:
        df              — датафрейм
        exclude_cols    — колонки які виключити з аналізу
        skew_threshold  — поріг для класифікації (default=0.5)
        log_threshold   — поріг |skewness| для обов'язкового логування (default=2.0)

    Returns:
        (result_df, right_log_cols, left_log_cols)
        right_log_cols — колонки БЕЗ від'ємних значень → log1p
        left_log_cols  — колонки З від'ємними значеннями → signed log1p
    """
    exclude  = set(exclude_cols or [])
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    rows = []
    for col in num_cols:
        series   = df[col].dropna()
        if len(series) < 2:
            continue

        skew      = float(series.skew())
        kurt      = float(series.kurtosis())
        null_pct  = df[col].isna().mean()
        has_neg   = bool((series < 0).any())      # ✅ від'ємні значення
        n_unique  = series.nunique()
        is_binary = n_unique <= 2                  # ✅ бінарні не логуємо

        if abs(skew) < skew_threshold:
            label = "✅ normal"
        elif abs(skew) < 1.0:
            label = "🟡 moderate"
        elif abs(skew) < 2.0:
            label = "🟠 high"
        else:
            label = "🔴 severe"

        direction = "right ▶" if skew > 0 else "left ◀"

        # ✅ needs_log: severe skew + не бінарна
        needs_log = abs(skew) >= log_threshold and not is_binary

        rows.append({
            "col":       col,
            "skewness":  round(skew, 4),
            "abs_skew":  round(abs(skew), 4),
            "kurtosis":  round(kurt, 4),
            "direction": direction,
            "null_pct":  round(null_pct, 3),
            "has_neg":   has_neg,
            "is_binary": is_binary,
            "label":     label,
            "needs_log": needs_log,
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("abs_skew", ascending=False)
        .reset_index(drop=True)
    )

    # ── Переліки для логування ────────────────────────────────
    # ✅ критерій — наявність від'ємних значень, а не знак skewness
    log_subset     = result[result["needs_log"]]
    right_log_cols = log_subset[~log_subset["has_neg"]]["col"].tolist()
    left_log_cols  = log_subset[log_subset["has_neg"]]["col"].tolist()

    # # ── Print ──────────────────────────────────────────────────
    # width = 95
    # print(f"\n{'═' * width}")
    # print(f"  {'col':<22} {'skewness':>10} {'kurtosis':>10} "
    #       f"{'direction':>10} {'null':>6} {'neg':>5}  {'label':<14} {'log?':>5}")
    # print(f"{'─' * width}")

    # for label_group in ["🔴 severe", "🟠 high", "🟡 moderate", "✅ normal"]:
    #     subset = result[result["label"] == label_group]
    #     if subset.empty:
    #         continue
    #     for _, r in subset.iterrows():
    #         log_marker = "✅" if r["needs_log"] else ""
    #         neg_marker = "✅" if r["has_neg"]   else ""
    #         print(
    #             f"  {r['col']:<22} {r['skewness']:>10.4f} {r['kurtosis']:>10.4f} "
    #             f"{r['direction']:>10} {r['null_pct']:>5.0%} {neg_marker:>5}  "
    #             f"{r['label']:<14} {log_marker:>5}"
    #         )
    #     print(f"{'─' * width}")

    # # ── Summary ───────────────────────────────────────────────
    # print(f"\n  Розподіл за рівнем skewness:")
    # for label_group in ["🔴 severe", "🟠 high", "🟡 moderate", "✅ normal"]:
    #     n = (result["label"] == label_group).sum()
    #     print(f"  {label_group:<15} : {n:>4} колонок")

    # print(f"\n  Поріг класифікації:")
    # print(f"  ✅ normal   : |skew| < {skew_threshold}")
    # print(f"  🟡 moderate : |skew| < 1.0")
    # print(f"  🟠 high     : |skew| < 2.0")
    # print(f"  🔴 severe   : |skew| ≥ 2.0")

    # print(f"\n  Колонки для логування (|skew| ≥ {log_threshold}, не бінарні):")
    # print(f"  right (без від'ємних) → log1p        : {len(right_log_cols):>4} колонок")
    # print(f"  left  (з від'ємними)  → signed log1p : {len(left_log_cols):>4} колонок")
    # print(f"  Всього                                : "
    #       f"{len(right_log_cols) + len(left_log_cols):>4} колонок")

    # if left_log_cols:
    #     print(f"\n  ⚠️  signed log1p колонки: {left_log_cols}")

    # print(f"{'═' * width}\n")

    return result, right_log_cols, left_log_cols



## =============== Final Batch Pipeline ===============

def add_card_features_with_train_map(
    df: pd.DataFrame,
    config: PreprocessConfig,
    card_stability_map=None,
    fit: bool = False,
):
    df = df.copy()

    for col in config.card_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # твоя оригінальна функція
    df = create_card_features(df, config.card_cols)

    if fit:
        card_stability_map = df.groupby("card1")["addr1"].nunique()

    if card_stability_map is not None and "card1" in df.columns:
        df["card_stability"] = (
            df["card1"]
            .map(card_stability_map)
            .fillna(1)
            .astype("float32")
        )

    return df, card_stability_map


def add_graph_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = create_composite_id(df, ["card1", "DeviceInfo"], "user_id")
    df = create_composite_id(df, ["DeviceInfo", "DeviceType"], "device_id")
    df = create_composite_id(df, ["addr1", "addr2"], "addr_id")
    df = create_composite_id(df, ["card1", "card2", "card3", "card5"], "card_id")

    return df


def add_behavioral_features(full_df):
    full_df = full_df.sort_values("TransactionDT").reset_index(drop=True)
    full_df["dt_prev"] = (
        full_df.groupby("user_id")["TransactionDT"]
        .diff()
        .fillna(1e6)
    )
    full_df["log_dt_prev"] = np.log1p(full_df["dt_prev"]).astype("float32")
    full_df["is_burst"] = (full_df["dt_prev"] < SECONDS_PER_HOUR).astype("int8")

    full_df = create_time_features(full_df)
    full_df = compute_tx_count_1h(full_df)
    full_df = compute_window_features(full_df, window=86400)
    full_df = velocity(full_df)
    full_df = create_rolling_features(full_df)
    full_df = compute_amt_zscore(full_df)

    return full_df


def make_xgb_matrix(df: pd.DataFrame, artifacts=None):
    drop_cols = ["user_id", "device_id", "addr_id", "card_id", "isFraud"]

    X = df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        errors="ignore",
    )

    y = df["isFraud"] if "isFraud" in df.columns else None

    if artifacts is not None:
        X = X.reindex(columns=artifacts.feature_cols, fill_value=0)
        cat_cols = artifacts.cat_cols
    else:
        cat_cols = [
            "addr1", "addr2", "DeviceInfo", "DeviceType",
            "card1", "card2", "card3", "card4", "card5", "card6",
        ]

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, y, cat_cols


def fit_preprocess_train_full(
    train_raw: pd.DataFrame,
    config: PreprocessConfig,
):
    exclude_from_num = set(
        config.cat_cols
        + config.node_source_cols
        + ["isFraud", "TransactionID"]
    )

    num_cols = [c for c in train_raw.columns if c not in exclude_from_num]

    train_df = drop_leaky_features(train_raw)
    train_df = train_df.sort_values("TransactionDT").reset_index(drop=True)

    train_df = d_columns_transformation(train_df)

    train_df, num_fill = fill_numeric_features(train_df, num_cols)

    cols_to_fill = set(config.cat_cols + config.node_source_cols)
    train_df, cat_fill = fill_categorical_features(train_df, cols_to_fill)

    train_df, card_stability_map = add_card_features_with_train_map(
        train_df, config, fit=True
    )

    train_df = add_graph_ids(train_df)

    train_df, encoders, encoded_cols = categorical_encoding_train(
        train_df,
        config.target_enc_cols,
        config.target_enc_num_cols,
        config.label_enc_cols,
        config.m_cols_binary,
        config.binary_id_cols,
        config.node_source_cols,
    )

    # Для train_full ratios fit/apply на самому train_full
    train_df, _ = create_amount_ratio_features(train_df, train_df.copy())

    is_missing_cols = [c for c in train_df.columns if c.endswith("_is_missing")]
    card_derived = ["card_stability", "card1_card2"]

    no_log_cols = (
        config.cat_cols
        + config.node_source_cols
        + card_derived
        + encoded_cols
        + is_missing_cols
        + [c for c in train_df.columns if c.endswith("_fraud_rate")]
        + ["user_id", "device_id", "addr_id", "card_id"]
        + ["TransactionID", "isFraud", "TransactionDT"]
        + ["amt_vs_user_mean", "amt_vs_card_mean"]
        + ["tx_per_day", "night_tx_ratio"]
        + ["how_sin", "how_cos", "hour_of_week"]
    )

    _, right_log_cols, left_log_cols = analyze_skewness(
        train_df,
        no_log_cols,
        skew_threshold=0.5,
        log_threshold=2.0,
    )

    train_df, log_artifacts = apply_log_transform(
        train_df,
        right_log_cols=right_log_cols,
        left_log_cols=left_log_cols,
        clip_quantile=0.999,
    )

    train_df = add_behavioral_features(train_df)
    train_df = train_df.drop(columns=["TransactionID"], errors="ignore")

    X_train_full, y_train_full, cat_cols = make_xgb_matrix(train_df)
    feature_cols = X_train_full.columns.tolist()

    artifacts = PreprocessArtifacts(
        num_fill=num_fill,
        cat_fill=cat_fill,
        encoders=encoders,
        encoded_cols=encoded_cols,
        log_artifacts=log_artifacts,
        card_stability_map=card_stability_map,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
    )

    return train_df, X_train_full, y_train_full, artifacts


def transform_test_batch(
    test_raw: pd.DataFrame,
    train_history_df: pd.DataFrame,
    config: PreprocessConfig,
    artifacts: PreprocessArtifacts,
):
    test_df = drop_leaky_features(test_raw)
    test_df = test_df.sort_values("TransactionDT").reset_index(drop=True)

    test_df = d_columns_transformation(test_df)

    test_df = apply_fill_values(
        test_df,
        artifacts.num_fill,
        artifacts.cat_fill,
    )

    test_df, _ = add_card_features_with_train_map(
        test_df,
        config,
        card_stability_map=artifacts.card_stability_map,
        fit=False,
    )

    test_df = add_graph_ids(test_df)

    test_df = categorical_encoding_valtest(
        test_df,
        artifacts.encoders,
    )

    _, test_df = create_amount_ratio_features(
        train_history_df,
        test_df,
    )

    test_df, _ = apply_log_transform(
        test_df,
        right_log_cols=artifacts.log_artifacts["right_log_cols"],
        left_log_cols=artifacts.log_artifacts["left_log_cols"],
        clip_values=artifacts.log_artifacts["clip_values"],
    )

    n_history = len(train_history_df)

    full_df = pd.concat(
        [train_history_df, test_df],
        ignore_index=True,
    ).sort_values("TransactionDT").reset_index(drop=True)

    full_df = add_behavioral_features(full_df)
    full_df = full_df.drop(columns=["TransactionID"], errors="ignore")

    test_df = full_df.iloc[n_history:].copy()

    X_test, _, _ = make_xgb_matrix(
        test_df,
        artifacts=artifacts,
    )

    return test_df, X_test


