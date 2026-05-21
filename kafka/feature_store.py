# feature_store.py

# Redis Feature Store — store real-time context

#   user:{user_id}:stats   → amt_mean, tx_count, last_dt
#   user:{user_id}:recent  → last 10 transactions (for temporal edges HGNN)
#   card:{card_id}:stats   → amt_mean per card (for amt_vs_card_mean)

import json
import logging
import redis
from config import REDIS_URL

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY  = 86400
TTL              = SECONDS_PER_DAY * 3   # 3 days
MAX_RECENT_USER_TXS = 50
MAX_RECENT_CARD_TXS = 10

class FeatureStore:
    """
    Redis Feature Store for real-time fraud detection.

    Stores:
      user:{user_id}:stats    -> user-level online statistics
      user:{user_id}:recent   -> recent transactions for user_id
      card:{card_id}:stats    -> card-level online statistics
      card:{card_id}:recent   -> recent transactions for card_id/Card_ID
    """

    def __init__(self, redis_url: str = REDIS_URL):
        self.client = redis.from_url(redis_url, decode_responses=True)
        logger.info(f"Redis connected: {redis_url}")
        print(f"Redis connected: {redis_url}")

    # User stats
    def get_user_stats(self, user_id: str) -> dict:
        key  = f"user:{user_id}:stats"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return {
            "amt_mean":   None,
            "amt_sum":    0.0,
            "tx_count":   0,
            "last_dt":    None,
            "first_dt": None,
            "tx_per_day": 1.0,
            "night_count": 0,
            "night_tx_ratio": 0.0,
        }

    def update_user_stats(self, user_id: str, tx: dict) -> None:
        """Update user statistics after transacton. """
        stats = self.get_user_stats(user_id)
        amt   = float(tx.get("TransactionAmt", 0) or 0)
        dt    = int(tx.get("TransactionDT", 0) or 0)

        old_count = int(stats.get("tx_count", 0))
        n            = old_count + 1
        old_mean     = float(stats.get("amt_mean") or amt)
        new_mean     = old_mean + (amt - old_mean) / n   # online mean update

        # tx_per_day
        first_dt = stats.get("first_dt") 
        first_dt = int(first_dt) if first_dt is not None else dt
        days     = max((dt - first_dt) / SECONDS_PER_DAY, 1.0)
        tx_per_day = n / days

        # night_tx_ratio
        hour       = (dt % SECONDS_PER_DAY) // 3600
        is_night   = 1 if 0 <= hour <= 5 else 0
        night_count = int(stats.get("night_count", 0)) + is_night
        night_ratio = night_count / n

        stats.update({
            "amt_mean":       round(new_mean, 4),
            "amt_sum":        round(float(stats.get("amt_sum", 0.0)) + amt, 4),
            "tx_count":       n,
            "last_dt":        dt,
            "first_dt":       first_dt,
            "tx_per_day":     round(tx_per_day, 4),
            "night_count":    night_count,
            "night_tx_ratio": round(night_ratio, 4),
        })

        self.client.setex(
            f"user:{user_id}:stats",
            TTL,
            json.dumps(stats),
        )

    # Card stats
    def get_card_stats(self, card_id: str) -> dict:
        key  = f"card:{card_id}:stats"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return {"amt_mean": None, "tx_count": 0,}

    def update_card_stats(self, card_id: str, tx: dict) -> None:
        """
        Update card statistics after transaction is processed. """
        stats = self.get_card_stats(card_id)
        amt = float(tx.get("TransactionAmt", 0) or 0)
        old_count = int(stats.get("tx_count", 0))
        n         = old_count + 1
        old_mean  = stats.get("amt_mean")
        old_mean = float(old_mean) if old_mean is not None else amt
        new_mean  = old_mean + (amt - old_mean) / n

        stats.update({
            "amt_mean": round(new_mean, 4),
            "tx_count": n,
        })
        self.client.setex(
            f"card:{card_id}:stats",
            TTL,
            json.dumps(stats),
        )

    @staticmethod
    def _slim_tx(tx: dict) -> dict:
        return {
            "TransactionID": tx.get("TransactionID"),
            "TransactionDT": tx.get("TransactionDT"),
            "TransactionAmt": tx.get("TransactionAmt"),
        }
    
    # Recent transactions (for temporal edges HGNN)
    def add_recent_user_transaction(self, user_id: str, tx: dict) -> None:
        """Store last 10 user's transactions. """
        key     = f"user:{user_id}:recent"
        self.client.lpush(key, json.dumps(self._slim_tx(tx)))
        self.client.ltrim(key, 0, MAX_RECENT_USER_TXS - 1)                    # 10 transactions
        self.client.expire(key, TTL)

    def get_recent_user_transactions(self, user_id: str, n: int = MAX_RECENT_USER_TXS) -> list:
        """Return last N user's transactions."""
        key  = f"user:{user_id}:recent"
        data = self.client.lrange(key, 0, n - 1)
        return [json.loads(d) for d in data]
    
    def add_recent_card_transaction(self, card_id: str, tx: dict) -> None:
        """
        Stores the most recent MAX_RECENT_CARD_TXS transactions for each card_id.
        Used for rolling amount features equivalent to batch Card_ID rolling features.
        """
        key = f"card:{card_id}:recent"
        self.client.lpush(key, json.dumps(self._slim_tx(tx)))
        self.client.ltrim(key, 0, MAX_RECENT_CARD_TXS - 1)
        self.client.expire(key, TTL)

    def get_recent_card_transactions(self, card_id: str, n: int = MAX_RECENT_CARD_TXS) -> list[dict]:
        key = f"card:{card_id}:recent"
        data = self.client.lrange(key, 0, n - 1)
        return [json.loads(d) for d in data]
    
    def get_context(self, user_id: str, card_id: str) -> dict:
        """
        Returns context expected by RealTimeFeatureEngine.
        """
        return {
            "user_stats": self.get_user_stats(user_id),
            "card_stats": self.get_card_stats(card_id),
            "recent_user_txs": self.get_recent_user_transactions(user_id),
            "recent_card_txs": self.get_recent_card_transactions(card_id),
        }

    def update_after_prediction(self, user_id: str, card_id: str, tx: dict) -> None:
        """
        Updates Redis state after prediction is produced.
        Important:
        Call this AFTER inference.
        """
        self.update_user_stats(user_id, tx)
        self.update_card_stats(card_id, tx)
        self.add_recent_user_transaction(user_id, tx)
        self.add_recent_card_transaction(card_id, tx)
