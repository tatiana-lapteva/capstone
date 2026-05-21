
# realtime_features.py

import math
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR / "ml" / "src"))
import data_preprocessing as data_prep

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY  = 86400
CARD_COLS = [
    "card1", "card2", "card3",
    "card4", "card5", "card6",]
CAT_COLS_FOR_CATEGORY = [
    "addr1", "addr2", "DeviceInfo", "DeviceType",
    "card1", "card2", "card3", "card4", "card5", "card6",]


class RealTimeFeatureEngine:
    """
    Apply batch-fitted preprocessing artifacts to one real-time transaction.
    Load once when consumer starts.
    Expected pipeline_artifacts:
    {
        "num_fill": dict,
        "cat_fill": dict,
        "encoders": dict,
        "log_artifacts": dict,
        "card_stability_map": dict / pd.Series,
        "user_amt_mean": dict / pd.Series,          optional
        "card_amt_mean": dict / pd.Series,          optional
        "global_tx_per_hour_map": dict / pd.Series, optional
        "global_tx_per_hour_default": float,        optional
    }
    """

    def __init__(self, pipeline_artifacts: dict, drop_cols: list = None):
        """
        pipeline_artifacts — dict from notebook:
          {num_fill, cat_fill, encoders, log_artifacts,
           card_stability_map, user_amt_mean, card_amt_mean}
        drop_cols — zero importance features
        """
        self.num_fill           = pipeline_artifacts["num_fill"]
        self.cat_fill           = pipeline_artifacts["cat_fill"]
        self.encoders           = pipeline_artifacts["encoders"]
        self.log_artifacts      = pipeline_artifacts["log_artifacts"]
        self.card_stability_map = pipeline_artifacts.get("card_stability_map", {})
        self.user_amt_mean      = pipeline_artifacts.get("user_amt_mean", {})
        self.card_amt_mean      = pipeline_artifacts.get("card_amt_mean", {})
        self.global_tx_per_hour_map = pipeline_artifacts.get("global_tx_per_hour_map", {})
        self.global_tx_per_hour_default = pipeline_artifacts.get("global_tx_per_hour_default", 0.0)
        self.drop_cols          = set(drop_cols or [])

        print(f"[DONE] RealTimeFeatureEngine loaded")
        print(f"   num_fill cols:    {len(self.num_fill)}")
        print(f"   cat_fill cols:    {len(self.cat_fill)}")
        print(f"   encoders:         {len(self.encoders)}")
        print(f"   drop_cols:        {len(self.drop_cols)}")


    def transform(self, tx: dict, redis_context: dict) -> pd.DataFrame:
        """
        Full preprocessing of one transaction.

        tx            — raw transaction from Kafka (dict)
        redis_context — context from Redis:
          {user_stats, card_stats,
          recent_user_txs, recent_card_txs}

        Return pd.DataFrame for XGBoost inference.
        """
        redis_context = redis_context or {}

        df = pd.DataFrame([tx])

        # Drop leaky features
        df = data_prep.drop_leaky_features(df)

        # D columns transformation
        df = self._d_columns_transformation(df)

        # Fill missing values (transform only)
        df = data_prep.apply_fill_values(df, self.num_fill, self.cat_fill)

        for col in CARD_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df = self._create_card_features_realtime(df)

        df = self._create_composite_ids(df)

        # Categorical encoding (transform only)
        df = data_prep.categorical_encoding_valtest(df, self.encoders)

        # Amount ratio features (з train maps)
        df = self._add_amount_ratio_features(df, redis_context)


        # Log transformation (transform only)
        df, _ = data_prep.apply_log_transform(
            df,
            right_log_cols = self.log_artifacts["right_log_cols"],
            left_log_cols  = self.log_artifacts["left_log_cols"],
            clip_values    = self.log_artifacts["clip_values"],
        )

        # Behavioral features from Redis
        df = self._add_behavioral_features(df, tx, redis_context)

        # Time features
        df = data_prep.create_time_features(df)

        # Drop zero importance features
        drop = [c for c in self.drop_cols if c in df.columns]
        if drop:
            df = df.drop(columns=drop)

        # Drop graph / target columns
        exclude = [
            "user_id", "device_id", "addr_id", "card_id",
            "isFraud", "TransactionID",
        ]
        df = df.drop(
            columns=[c for c in exclude if c in df.columns],
            errors="ignore",
        )

        for col in CAT_COLS_FOR_CATEGORY:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df


    def _d_columns_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        """
        df = df.copy()
        d_cols = [
            c for c in df.columns
            if c.startswith("D") and c[1:].isdigit()
        ]
        for col in d_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["DT_day"] = (
            df["TransactionDT"] / SECONDS_PER_DAY
        ).astype("float32")

        for col in d_cols:
            df[col] = (df["DT_day"] - df[col]).astype("float32")

        if "D1" in df.columns:
            df["D1"] = df["D1"].round(0).astype("float32")

        return df


    def _create_card_features_realtime(
            self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Card features without isFraud.
        card_stability from train map.
        """
        df = df.copy()

        # card1_card2
        if "card1" in df.columns and "card2" in df.columns:
            df["card1_card2"] = (
                df["card1"].astype(str) + "_" + df["card2"].astype(str)
            )

        # card_stability з train map (no leakage)
        if "card1" in df.columns:
            df["card_stability"] = (
                df["card1"]
                .astype(str)
                .map(self.card_stability_map)
                .fillna(1.0)
                .astype("float32")
            )
        else:
            df["card_stability"] = 1.0

        return df


    def _create_composite_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite IDs for entity nodes."""
        df = df.copy()

        def safe_composite_id(data, cols, id_name):
            available = [c for c in cols if c in data.columns]
            if not available:
                data[id_name] = "unknown"
                return data
            result = (
                data[available[0]]
                .astype(str)
                .replace("nan", "unknown")
                .replace("<NA>", "unknown"))
                         
            for col in available[1:]:
                result = (result + "_" + data[col].astype(str).replace("nan", "unknown").replace("<NA>", "unknown"))
            data[id_name] = result
            return data
        
        df = safe_composite_id(df, ["card1", "DeviceInfo"], "user_id")
        df = safe_composite_id(df, ["DeviceInfo", "DeviceType"], "device_id")
        df = safe_composite_id(df, ["addr1", "addr2"], "addr_id")
        df = safe_composite_id(df, ["card1", "card2", "card3", "card5"], "card_id")

        return df


    def _add_amount_ratio_features(
            self,
            df:            pd.DataFrame,
            redis_context: dict,
    ) -> pd.DataFrame:
        """
        amt_vs_user_mean і amt_vs_card_mean.
        Пріоритет: Redis stats → train map → amt самої транзакції.
        """
        df  = df.copy()
        amt = float(df["TransactionAmt"].iloc[0]) if "TransactionAmt" in df.columns else 0.0

        # user mean: Redis (онлайн) або train map (offline)
        user_id      = df["user_id"].iloc[0] if "user_id" in df.columns else None
        card1 = str(df["card1"].iloc[0]) if "card1" in df.columns else None

        user_stats   = redis_context.get("user_stats", {})
        card_stats   = redis_context.get("card_stats", {})

        user_mean_redis = user_stats.get("amt_mean")
        user_mean_train = self.user_amt_mean.get(user_id, None) if user_id is not None else None
        user_mean       = self._first_valid_number(user_mean_redis, user_mean_train, amt)
        
        card_mean_redis = card_stats.get("amt_mean")
        card_mean_train = self.card_amt_mean.get(card1, None) if card1 is not None else None
        card_mean       = self._first_valid_number(card_mean_redis, card_mean_train, amt)

        df["amt_vs_user_mean"] = np.clip(
            amt / (float(user_mean) + 1e-8),
            0,
            100,
        ).astype("float32") if hasattr(np.clip(amt / (float(user_mean) + 1e-8), 0, 100), "astype") else float(
            np.clip(amt / (float(user_mean) + 1e-8), 0, 100)
        )

        df["amt_vs_card_mean"] = np.clip(
            amt / (float(card_mean) + 1e-8),
            0,
            100,
        ).astype("float32") if hasattr(np.clip(amt / (float(card_mean) + 1e-8), 0, 100), "astype") else float(
            np.clip(amt / (float(card_mean) + 1e-8), 0, 100)
        )

        return df


    def _add_behavioral_features(
            self,
            df:            pd.DataFrame,
            tx:            dict,
            redis_context: dict,
    ) -> pd.DataFrame:
        """
        Behavioral features from Redis context.
        """
        df    = df.copy()
        dt    = int(tx.get("TransactionDT", 0) or 0)
        amt   = float(tx.get("TransactionAmt", 0) or 0)

        user_stats  = redis_context.get("user_stats", {})
        recent_user_txs = redis_context.get("recent_user_txs", redis_context.get("recent_txs", []))
        recent_card_txs = redis_context.get("recent_card_txs", recent_user_txs)
        recent_user_txs = self._sort_recent_txs(recent_user_txs)
        recent_card_txs = self._sort_recent_txs(recent_card_txs)


        # dt_prev
        # Batch: full_df.groupby("user_id")["TransactionDT"].diff()
        # Real-time: last_dt from Redis
        last_dt          = user_stats.get("last_dt")

        if last_dt is None:
            dt_prev = 1_000_000
        else:
            dt_prev = max(dt - int(last_dt), 1)

        df["dt_prev"]    = float(dt_prev)
        df["log_dt_prev"] = float(math.log1p(dt_prev))
        df["is_burst"]   = int(dt_prev < SECONDS_PER_HOUR)

        # tx_count_1h
        # Real-time: compute from recent_txs in Redis
        count_1h_hist = sum(
            1 for r in recent_user_txs
            if 0 <= dt - int(r.get("TransactionDT", 0) or 0) <= SECONDS_PER_HOUR
        )
        count_1h = count_1h_hist + 1

        df["tx_count_1h"] = int(count_1h)
        df["log_tx_count"] = float(math.log1p(count_1h))
        df["tx_velocity"] = float(
            count_1h / max(dt_prev / SECONDS_PER_HOUR, 1e-8)
        )

        # tx_count_24h, amt_sum_24h, amt_mean_24h
        window_24h = [
            r for r in recent_user_txs
            if 0 <= dt - int(r.get("TransactionDT", 0) or 0) <= SECONDS_PER_DAY
        ]
        amts_24h = [float(r.get("TransactionAmt", 0) or 0) for r in window_24h]
        df["tx_count_24h"] = int(len(window_24h))
        df["amt_sum_24h"]  = float(sum(amts_24h))
        df["amt_mean_24h"] = float(
            sum(amts_24h) / len(amts_24h) if amts_24h else 0.0
        )

        # rolling features з recent_txs
        # Batch: create_rolling_features — shift(1).rolling(10)
        # Real-time: from last 10 transactions in Redis
        prev_amts = [float(r.get("TransactionAmt", 0) or 0) for r in recent_card_txs[:10]]

        if prev_amts:
            rolling_mean = float(np.mean(prev_amts))
            rolling_std  = float(np.std(prev_amts, ddof=1)) if len(prev_amts) > 1 else -1.0
            rolling_min  = float(np.min(prev_amts))
            rolling_max  = float(np.max(prev_amts))
            no_history = 0
        else:
            rolling_mean = amt
            rolling_std  = -1.0
            rolling_min  = amt
            rolling_max  = amt
            no_history = 1

        df["amt_ratio_mean"]  = float(amt / (rolling_mean + 1.0))
        df["amt_diff_mean"]   = float(amt - rolling_mean)
        df["amt_rolling_std"] = rolling_std
        df["amt_rolling_min"] = rolling_min
        df["amt_rolling_max"] = rolling_max
        df["no_history"]      = int(no_history)

        # amt_zscore
        # Batch: compute_amt_zscore
        if rolling_std > 0:
            df["amt_zscore"] = float(
                np.clip((amt - df["amt_mean_24h"].iloc[0]) / (rolling_std + 1e-8), -10, 10)
            )
        else:
            df["amt_zscore"] = 0.0

        # tx_per_day, night_tx_ratio
        # Batch: create_transaction_level_features
        # Real-time: Redis user_stats
        df["tx_per_day"]     = float(user_stats.get("tx_per_day", 1.0))
        df["night_tx_ratio"] = float(user_stats.get("night_tx_ratio", 0.0))

        # hour_of_week + cyclic
        hour_of_week         = (dt // 3600) % 168
        df["hour_of_week"]   = hour_of_week
        df["how_sin"]        = float(math.sin(2 * math.pi * hour_of_week / 168))
        df["how_cos"]        = float(math.cos(2 * math.pi * hour_of_week / 168))

        # Global transactions per hour from train-fitted map, not user_stats
        hour_bucket = int(dt // 3600)
        global_rate = self.global_tx_per_hour_map.get(
            hour_bucket,
            self.global_tx_per_hour_default,
        )
        df["global_tx_per_hour"] = float(global_rate)

        return df
    
    @staticmethod
    def _sort_recent_txs(recent_txs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Sort previous transactions by TransactionDT descending.
        Current transaction must not be included.
        """
        return sorted(
            recent_txs or [],
            key=lambda r: int(r.get("TransactionDT", 0) or 0),
            reverse=True,
        )

    @staticmethod
    def _first_valid_number(*values: Any) -> float:
        """
        Returns first non-null, finite numeric value.
        """
        for value in values:
            if value is None:
                continue
            try:
                value = float(value)
                if np.isfinite(value):
                    return value
            except (TypeError, ValueError):
                continue
        return 0.0