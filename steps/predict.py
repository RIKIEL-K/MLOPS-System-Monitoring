"""
steps/predict.py — Classe Predictor

Charge le modèle (pickle local ou MLflow Registry) et expose
evaluate() pour scorer le jeu de test, et predict() pour l'API FastAPI.
"""

import logging
import os
import pickle

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class Predictor:
    """
    Évalue le modèle sur le jeu de test et expose la prédiction pour l'API.

    Usage:
        predictor = Predictor()
        results = predictor.evaluate(test_data)
        predictions_df = predictor.predict(["log message 1", "log message 2"])
    """

    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._model_data: dict | None = None

    # ── API publique ────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Évaluer le modèle sur un jeu de test et retourner les métriques clés.

        Métriques retournées (clustering non supervisé) :
          - silhouette_score : cohésion des clusters (plus proche de 1 = mieux)
          - anomaly_rate     : part des logs dans des clusters "Erreurs" (0–1)
          - n_clusters       : nombre de clusters identifiés
          - cluster_distribution : nombre de logs par cluster

        Args:
            df: DataFrame nettoyé avec la colonne `log_pattern`.

        Returns:
            Dict de métriques.
        """
        model_data = self._load_model()
        vectorizer     = model_data["vectorizer"]
        kmeans         = model_data["kmeans"]
        cluster_labels = model_data["cluster_labels"]

        X      = vectorizer.transform(df["log_pattern"])
        preds  = kmeans.predict(X)
        labels = [cluster_labels.get(int(p), "Unknown") for p in preds]

        df = df.copy()
        df["cluster_id"]    = preds
        df["cluster_label"] = labels

        # Silhouette score (nécessite au moins 2 clusters et 2 labels uniques)
        try:
            sil_score = float(silhouette_score(X, preds))
        except ValueError:
            sil_score = 0.0
            logger.warning("Impossible de calculer le silhouette score (clusters insuffisants).")

        # Anomaly rate : proportion de logs dans des clusters "Erreurs"
        error_mask   = df["cluster_label"].str.contains("Erreurs", case=False, na=False)
        anomaly_rate = float(error_mask.mean()) if len(df) > 0 else 0.0

        # Distribution par cluster
        cluster_dist = (
            df.groupby("cluster_label").size()
            .sort_values(ascending=False)
            .to_dict()
        )

        results = {
            "silhouette_score":       round(sil_score, 4),
            "anomaly_rate":           round(anomaly_rate, 4),
            "n_clusters":             kmeans.n_clusters,
            "n_samples_evaluated":    len(df),
            "cluster_distribution":   cluster_dist,
        }

        logger.info(
            "Évaluation — silhouette=%.3f | anomaly_rate=%.2f%% | n_clusters=%d",
            sil_score, anomaly_rate * 100, kmeans.n_clusters,
        )
        return results

    def predict(self, messages: list | pd.DataFrame) -> pd.DataFrame:
        """
        Prédire le cluster de messages de logs (pour l'API FastAPI).

        Args:
            messages: Liste de chaînes ou DataFrame avec colonne `message`.

        Returns:
            DataFrame avec `cluster_id` et `cluster_label`.
        """
        from steps.clean import Cleaner

        model_data     = self._load_model()
        vectorizer     = model_data["vectorizer"]
        kmeans         = model_data["kmeans"]
        cluster_labels = model_data["cluster_labels"]

        if isinstance(messages, pd.DataFrame):
            msg_list = messages["message"].tolist()
        elif isinstance(messages, list):
            msg_list = messages
        else:
            msg_list = [str(messages)]

        cleaned = [Cleaner.clean_message_for_inference(m) for m in msg_list]
        X_pred  = vectorizer.transform(cleaned)
        preds   = kmeans.predict(X_pred)
        labels  = [cluster_labels.get(int(p), "Unknown") for p in preds]

        return pd.DataFrame({"cluster_id": preds.tolist(), "cluster_label": labels})

    # ── Private ────────────────────────────────────────────────────────

    def _load_model(self) -> dict:
        """Charger le modèle depuis pickle (lazy loading avec cache)."""
        if self._model_data is not None:
            return self._model_data

        pickle_path = os.path.join(
            self.config["paths"]["models_dir"],
            "log_clustering_model.pkl",
        )

        if not os.path.isfile(pickle_path):
            raise FileNotFoundError(
                f"Modèle introuvable : {pickle_path}\n"
                "  → Lancez `python main.py` pour entraîner le modèle d'abord."
            )

        with open(pickle_path, "rb") as f:
            self._model_data = pickle.load(f)

        logger.info("Modèle chargé depuis : %s", pickle_path)
        return self._model_data
