"""
steps/train.py — Classe Trainer

Entraîne un pipeline TF-IDF + K-Means et le logue dans MLflow local.
Lit les hyperparamètres depuis params.yaml (DVC) et les chemins depuis config.yml.
Exporte metrics.json et plots/cluster_distribution.csv pour DVC.
"""

import json
import logging
import os
import pickle
import re
import tempfile

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models.signature import infer_signature
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class Trainer:
    """
    Entraîne TF-IDF + K-Means, logue dans MLflow local,
    et exporte les métriques/plots pour DVC.

    Usage:
        trainer = Trainer()
        trainer.train(train_data)
        trainer.save_model()
    """

    def __init__(self, config_path: str = "config.yml",
                 params_path: str = "params.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        with open(params_path, "r", encoding="utf-8") as f:
            self.params = yaml.safe_load(f)

        mlflow.set_tracking_uri(self.cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.params["mlflow"]["experiment_name"])

        self.model_name    = self.params["mlflow"]["model_name"]
        self.models_dir    = self.cfg["paths"]["models_dir"]
        self.metrics_file  = self.cfg["paths"]["metrics_file"]
        self.plots_dir     = self.cfg["paths"]["plots_dir"]
        self.artifact_path = self.cfg["mlflow"]["artifact_path"]

        # Remplis après train()
        self.vectorizer:     TfidfVectorizer | None = None
        self.kmeans:         KMeans | None = None
        self.cluster_labels: dict  = {}
        self.feature_names:  list  = []
        self.run_id:         str   = ""
        self.silhouette:     float = 0.0
        self.anomaly_rate:   float = 0.0

    # ── API publique ────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> str:
        """
        Entraîner le pipeline TF-IDF + K-Means.

        Args:
            df: DataFrame nettoyé avec la colonne `log_pattern`.

        Returns:
            run_id MLflow.
        """
        p_tfidf  = self.params["tfidf"]
        p_kmeans = self.params["kmeans"]

        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            logger.info("MLflow run démarré : %s", self.run_id)

            mlflow.log_param("n_samples", len(df))

            # ── TF-IDF ─────────────────────────────────────────────────
            self.vectorizer = TfidfVectorizer(
                max_features=p_tfidf["max_features"],
                stop_words="english",
                min_df=p_tfidf["min_df"],
                max_df=p_tfidf["max_df"],
                token_pattern=r"(?u)\b\w+\b",
            )
            X = self.vectorizer.fit_transform(df["log_pattern"])
            self.feature_names = self.vectorizer.get_feature_names_out().tolist()
            logger.info("TF-IDF : %d docs × %d features", X.shape[0], X.shape[1])

            mlflow.log_params({
                "max_features": p_tfidf["max_features"],
                "min_df":       p_tfidf["min_df"],
                "max_df":       p_tfidf["max_df"],
            })
            mlflow.log_metric("tfidf_vocabulary_size", len(self.feature_names))

            # ── Évaluation k-range ──────────────────────────────────────
            n_init       = p_kmeans["n_init"]
            random_state = p_kmeans["random_state"]
            k_values     = [int(k.strip()) for k in p_kmeans["k_range"].split(",")]
            best_k, best_sil = self._evaluate_k_range(X, k_values, n_init, random_state)
            mlflow.log_metric("best_k_silhouette",    best_k)
            mlflow.log_metric("best_silhouette_score", round(best_sil, 4))

            # ── K-Means final ───────────────────────────────────────────
            n_clusters   = p_kmeans["n_clusters"]
            self.kmeans  = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
            labels       = self.kmeans.fit_predict(X)
            self.silhouette = float(silhouette_score(X, labels))
            df = df.copy()
            df["cluster_id"] = labels

            mlflow.log_params({
                "n_clusters":   n_clusters,
                "n_init":       n_init,
                "random_state": random_state,
                "k_range":      p_kmeans["k_range"],
            })
            mlflow.log_metric("silhouette_score", round(self.silhouette, 4))
            mlflow.log_metric("inertia",          round(self.kmeans.inertia_, 2))
            logger.info("K-Means : silhouette=%.3f | inertia=%.0f",
                        self.silhouette, self.kmeans.inertia_)

            # ── Labeling ────────────────────────────────────────────────
            df, self.cluster_labels, top_words = self._label_clusters(df)
            for cid, label in self.cluster_labels.items():
                mlflow.log_param(f"cluster_{cid}_label", label)
                logger.info("  Cluster %d : %s", cid, label)

            error_mask        = df["cluster_label"].str.contains("Erreurs", case=False, na=False)
            self.anomaly_rate = float(error_mask.mean()) if len(df) > 0 else 0.0
            mlflow.log_metric("anomaly_rate", round(self.anomaly_rate, 6))

            # ── Résumé CSV ──────────────────────────────────────────────
            summary_path = self._export_summary(df, top_words)
            mlflow.log_artifact(summary_path, "outputs")

            # ── Modèle MLflow ───────────────────────────────────────────
            self._log_mlflow_model()
            logger.info("Modèle enregistré dans MLflow Registry : %s", self.model_name)

        return self.run_id

    def save_model(self) -> str:
        """Sauvegarder le modèle en pickle local pour le serving."""
        if self.vectorizer is None or self.kmeans is None:
            raise RuntimeError("Appelez train() avant save_model().")

        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, "log_clustering_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "vectorizer":     self.vectorizer,
                "kmeans":         self.kmeans,
                "cluster_labels": self.cluster_labels,
                "feature_names":  self.feature_names,
            }, f)
        logger.info("Modèle sauvegardé : %s", model_path)
        return model_path

    def save_metrics(self, extra: dict | None = None) -> str:
        """
        Exporter metrics.json pour DVC (dvc metrics show / dvc exp run).

        Args:
            extra: Métriques additionnelles à fusionner (ex: depuis Predictor).

        Returns:
            Chemin du fichier metrics.json.
        """
        metrics = {
            "silhouette_score": round(self.silhouette, 4),
            "anomaly_rate":     round(self.anomaly_rate, 4),
            "inertia":          round(self.kmeans.inertia_, 2) if self.kmeans else 0,
            "n_clusters":       self.kmeans.n_clusters if self.kmeans else 0,
            "tfidf_vocab_size": len(self.feature_names),
        }
        if extra:
            metrics.update(extra)

        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Métriques DVC exportées : %s", self.metrics_file)
        return self.metrics_file

    def save_plots(self, cluster_distribution: dict) -> str:
        """
        Exporter plots/cluster_distribution.csv pour dvc plots.

        Args:
            cluster_distribution: Dict {label: count} depuis Predictor.evaluate().

        Returns:
            Chemin du fichier CSV.
        """
        os.makedirs(self.plots_dir, exist_ok=True)
        plot_path = os.path.join(self.plots_dir, "cluster_distribution.csv")
        rows = [
            {"cluster_label": label, "count": count}
            for label, count in cluster_distribution.items()
        ]
        pd.DataFrame(rows).to_csv(plot_path, index=False)
        logger.info("Plots DVC exportés : %s", plot_path)
        return plot_path

    # ── Private ────────────────────────────────────────────────────────

    def _evaluate_k_range(self, X, k_values, n_init, random_state):
        silhouettes = []
        for k in k_values:
            km  = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            pred = km.fit_predict(X)
            sil  = silhouette_score(X, pred)
            silhouettes.append(sil)
            logger.debug("  k=%2d → silhouette=%.3f", k, sil)
        best_idx = int(np.argmax(silhouettes))
        logger.info("Meilleur k : %d (silhouette=%.3f)", k_values[best_idx], silhouettes[best_idx])
        return k_values[best_idx], silhouettes[best_idx]

    def _label_clusters(self, df):
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        top_words: dict = {}
        cluster_labels: dict = {}

        for cid in range(self.kmeans.n_clusters):
            top_terms    = [self.feature_names[i] for i in order_centroids[cid, :10]]
            top_words[cid] = top_terms
            cluster_data = df[df["cluster_id"] == cid]

            top_component = cluster_data["component"].mode().iloc[0] if len(cluster_data) > 0 else "unknown"
            top_endpoint  = cluster_data["endpoint"].mode().iloc[0]  if len(cluster_data) > 0 else "unknown"

            error_flag = any("500" in t or "502" in t or "503" in t for t in top_terms[:5])
            warn_flag  = any("400" in t or "401" in t or "403" in t for t in top_terms[:5])

            if error_flag:
                label = f"Erreurs Serveur ({top_component})"
            elif warn_flag:
                label = f"Accès Non Autorisés ({top_component})"
            else:
                label = f"Opérations {top_component.capitalize()} ({top_endpoint})"

            cluster_labels[cid] = label

        df = df.copy()
        df["cluster_label"] = df["cluster_id"].map(cluster_labels)
        return df, cluster_labels, top_words

    def _export_summary(self, df, top_words):
        rows = []
        for cid in range(self.kmeans.n_clusters):
            cd = df[df["cluster_id"] == cid]
            rows.append({
                "ClusterID":       cid,
                "Label":           self.cluster_labels.get(cid, "Unknown"),
                "TopTerms":        ", ".join(top_words[cid][:8]),
                "ExampleLog":      cd["message"].iloc[0] if len(cd) > 0 else "",
                "Count":           int(len(cd)),
                "Percentage":      round(len(cd) / len(df) * 100, 1),
                "AvgResponseTime": round(cd["response_time_ms"].mean(), 1),
            })
        summary      = pd.DataFrame(rows).sort_values("Count", ascending=False)
        tmp_dir      = tempfile.mkdtemp()
        summary_path = os.path.join(tmp_dir, "log_clusters_summary.csv")
        summary.to_csv(summary_path, index=False)
        return summary_path

    def _log_mlflow_model(self):
        input_example  = pd.DataFrame({"message": ["level=error msg='connection refused'"]})
        output_example = pd.DataFrame({"cluster_id": [0], "cluster_label": ["Erreurs Serveur"]})
        signature      = infer_signature(input_example, output_example)
        mlflow.pyfunc.log_model(
            artifact_path=self.artifact_path,
            python_model=_LogClusteringModel(self.vectorizer, self.kmeans, self.cluster_labels),
            signature=signature,
            input_example=input_example,
            registered_model_name=self.model_name,
        )


# ── Modèle PyFunc ──────────────────────────────────────────────────────────────

class _LogClusteringModel(mlflow.pyfunc.PythonModel):
    def __init__(self, vectorizer, kmeans, cluster_labels):
        self.vectorizer     = vectorizer
        self.kmeans         = kmeans
        self.cluster_labels = cluster_labels

    def _clean(self, msg):
        if not isinstance(msg, str):
            return ""
        msg = re.sub(r"\b\d+\b", "<NUM>", msg)
        msg = re.sub(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
                     "<UUID>", msg, flags=re.IGNORECASE)
        msg = re.sub(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "<IP>", msg)
        return msg.lower().strip()

    def predict(self, context, model_input):
        import pandas as _pd
        if isinstance(model_input, _pd.DataFrame):
            col      = "message" if "message" in model_input.columns else model_input.columns[0]
            messages = model_input[col].tolist()
        elif isinstance(model_input, list):
            messages = model_input
        else:
            messages = [str(model_input)]
        cleaned = [self._clean(m) for m in messages]
        X_pred  = self.vectorizer.transform(cleaned)
        preds   = self.kmeans.predict(X_pred)
        labels  = [self.cluster_labels.get(int(p), "Unknown") for p in preds]
        return _pd.DataFrame({"cluster_id": preds.tolist(), "cluster_label": labels})
