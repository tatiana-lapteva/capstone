# kafka/explainability.py

import numpy as np
import pandas as pd


class FraudExplainer:
    """
    Lightweight explanation layer for real-time fraud predictions.

    Uses:
      - XGBoost feature importance / contribution if available
      - semantic feature names for HGNN embeddings
      - rule-based interpretation for human-readable explanations
    """

    def __init__(
        self,
        model,
        feature_cols: list[str],
        emb_cols: list[str],
        top_k: int = 10,
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.emb_cols = set(emb_cols)
        self.top_k = top_k

    def explain(self, X: pd.DataFrame, fraud_score: float, threshold: float) -> dict:
        """
        Returns structured explanation for one prediction.
        """

        top_features = self._get_top_xgb_contributions(X)

        tabular_reasons = []
        graph_reasons = []

        for item in top_features:
            feature = item["feature"]

            if feature in self.emb_cols:
                graph_reasons.append(item)
            else:
                tabular_reasons.append(item)

        human_reasons = self._human_readable_reasons(
            X=X,
            fraud_score=fraud_score,
            threshold=threshold,
            tabular_reasons=tabular_reasons,
            graph_reasons=graph_reasons,
        )

        return {
            "decision": "fraud" if fraud_score > threshold else "non_fraud",
            "fraud_score": round(float(fraud_score), 6),
            "threshold": threshold,
            "top_features": top_features,
            "top_tabular_features": tabular_reasons[: self.top_k],
            "top_graph_embedding_features": graph_reasons[: self.top_k],
            "human_readable_reasons": human_reasons,
        }

    def _get_top_xgb_contributions(self, X: pd.DataFrame) -> list[dict]:
        """
        Uses XGBoost pred_contribs if possible.
        Works with xgboost.XGBClassifier and Booster-compatible models.
        """

        try:
            booster = self.model.get_booster()
            contribs = booster.predict(
                self._to_dmatrix(X),
                pred_contribs=True,
            )[0]

            # Last value is bias term
            feature_contribs = contribs[:-1]

            rows = []
            for feature, value, contribution in zip(
                X.columns,
                X.iloc[0].values,
                feature_contribs,
            ):
                rows.append({
                    "feature": feature,
                    "value": self._safe_value(value),
                    "contribution": round(float(contribution), 6),
                    "direction": "increases_fraud_risk"
                    if contribution > 0
                    else "decreases_fraud_risk",
                })

            rows = sorted(
                rows,
                key=lambda x: abs(x["contribution"]),
                reverse=True,
            )

            return rows[: self.top_k]

        except Exception:
            return self._fallback_feature_importance(X)

    def _to_dmatrix(self, X: pd.DataFrame):
        import xgboost as xgb

        return xgb.DMatrix(
            X,
            enable_categorical=True,
        )

    def _fallback_feature_importance(self, X: pd.DataFrame) -> list[dict]:
        """
        Fallback if pred_contribs is unavailable.
        This is less precise: uses global feature importance only.
        """

        try:
            importances = self.model.feature_importances_
        except Exception:
            return []

        rows = []
        for feature, value, importance in zip(
            X.columns,
            X.iloc[0].values,
            importances,
        ):
            if importance <= 0:
                continue

            rows.append({
                "feature": feature,
                "value": self._safe_value(value),
                "contribution": round(float(importance), 6),
                "direction": "important_feature_global",
            })

        rows = sorted(
            rows,
            key=lambda x: abs(x["contribution"]),
            reverse=True,
        )

        return rows[: self.top_k]

    def _human_readable_reasons(
        self,
        X: pd.DataFrame,
        fraud_score: float,
        threshold: float,
        tabular_reasons: list[dict],
        graph_reasons: list[dict],
    ) -> list[str]:
        reasons = []

        row = X.iloc[0]

        if fraud_score > threshold:
            reasons.append(
                "The transaction was classified as fraud because its predicted risk score is above the selected threshold."
            )
        else:
            reasons.append(
                "The transaction was classified as non-fraud because its predicted risk score is below the selected threshold."
            )

        if "is_burst" in row and float(row["is_burst"]) == 1:
            reasons.append(
                "The transaction occurred shortly after a previous transaction from the same user, indicating burst-like behavior."
            )

        if "log_dt_prev" in row:
            reasons.append(
                "The time gap from the previous user transaction was considered by the model."
            )

        if "amt_vs_user_mean" in row and float(row["amt_vs_user_mean"]) > 2:
            reasons.append(
                "The transaction amount is significantly higher than the user's historical average amount."
            )

        if "amt_vs_card_mean" in row and float(row["amt_vs_card_mean"]) > 2:
            reasons.append(
                "The transaction amount is significantly higher than the card's historical average amount."
            )

        fraud_rate_features = [
            "card1_fraud_rate",
            "DeviceInfo_fraud_rate",
            "addr1_fraud_rate",
            "card_id_fraud_rate",
        ]

        for col in fraud_rate_features:
            if col in row and float(row[col]) > 0.1:
                reasons.append(
                    f"The feature {col} indicates elevated historical fraud risk for the related entity."
                )

        if graph_reasons:
            reasons.append(
                "Graph-based HGNN embeddings contributed to the decision, meaning the user's/device's/card's relational context influenced the prediction."
            )

        if tabular_reasons:
            top = tabular_reasons[0]
            reasons.append(
                f"The strongest tabular contributor was {top['feature']} with value {top['value']}."
            )

        if graph_reasons:
            top_g = graph_reasons[0]
            reasons.append(
                f"The strongest graph embedding contributor was {top_g['feature']}."
            )

        return reasons

    @staticmethod
    def _safe_value(value):
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if pd.isna(value):
            return None
        return value