"""
steps/train.py — Entraînement TF-IDF + K-Means avec MLflow local

Ce module est appelé par main.py. Il n'utilise pas Kubeflow ni MinIO.
MLflow est configuré localement (./mlruns).

Flux :
  1. Charger config.yml
  2. Ingérer et nettoyer les données (via steps/ingest et steps/clean)
  3. Vectoriser (TF-IDF)
  4. Évaluer la plage de k (elbow + silhouette)
  5. Entraîner K-Means final
  6. Logger tout dans MLflow local
  7. Sauvegarder le modèle PyFunc dans models/
"""

import os
import re
import tempfile
import warnings

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

warnings.filterwarnings("ignore")


def load_config(config_path: str = "config.yml") -> dict:
    """Charger la configuration depuis config.yml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def label_clusters(df: pd.DataFrame, kmeans_model: KMeans,
                   feature_names: list) -> tuple[pd.DataFrame, dict, dict]:
    """
    Attribuer un label interprétatif à chaque cluster.

    Analyse les top-termes TF-IDF de chaque centroïde et déduit
    si le cluster correspond à des erreurs, des accès non autorisés,
    ou des opérations normales.

    Args:
        df:            DataFrame avec colonne `cluster_id`.
        kmeans_model:  Modèle K-Means entraîné.
        feature_names: Liste des features TF-IDF.

    Returns:
        (df_enrichi, cluster_labels, top_words)
    """
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
    top_words: dict = {}
    cluster_labels: dict = {}

    for cid in range(kmeans_model.n_clusters):
        top_terms = [feature_names[idx] for idx in order_centroids[cid, :10]]
        top_words[cid] = top_terms
        cluster_data = df[df["cluster_id"] == cid]

        top_component = (
            cluster_data["component"].mode().iloc[0]
            if len(cluster_data) > 0 else "unknown"
        )
        top_endpoint = (
            cluster_data["endpoint"].mode().iloc[0]
            if len(cluster_data) > 0 else "unknown"
        )

        error_flag = any("500" in t or "502" in t or "503" in t for t in top_terms[:5])
        warn_flag = any("400" in t or "401" in t or "403" in t for t in top_terms[:5])

        if error_flag:
            label = f"Erreurs Serveur ({top_component})"
        elif warn_flag:
            label = f"Accès Non Autorisés ({top_component})"
        else:
            label = f"Opérations {top_component.capitalize()} ({top_endpoint})"

        cluster_labels[cid] = label

    df["cluster_label"] = df["cluster_id"].map(cluster_labels)
    return df, cluster_labels, top_words



class LogClusteringPipelineModel(mlflow.pyfunc.PythonModel):
    """
    Modèle MLflow PyFunc qui encapsule TF-IDF + K-Means.

    Accepte un DataFrame avec une colonne `message` (texte brut)
    et retourne un DataFrame avec `cluster_id` et `cluster_label`.
    Compatible avec `mlflow models serve` et l'API FastAPI.
    """

    def __init__(self, vectorizer: TfidfVectorizer, kmeans: KMeans,
                 cluster_labels: dict):
        self.vectorizer = vectorizer
        self.kmeans = kmeans
        self.cluster_labels = cluster_labels

    def _clean_message(self, msg: str) -> str:
        if not isinstance(msg, str):
            return ""
        msg = re.sub(r"\b\d+\b", "<NUM>", msg)
        msg = re.sub(
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
            "<UUID>", msg, flags=re.IGNORECASE
        )
        msg = re.sub(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "<IP>", msg)
        return msg.lower().strip()

    def predict(self, context, model_input):
        import pandas as _pd
        if isinstance(model_input, _pd.DataFrame):
            col = "message" if "message" in model_input.columns else model_input.columns[0]
            messages = model_input[col].tolist()
        elif isinstance(model_input, list):
            messages = model_input
        else:
            messages = [str(model_input)]

        cleaned = [self._clean_message(m) for m in messages]
        X_pred = self.vectorizer.transform(cleaned)
        preds = self.kmeans.predict(X_pred)
        labels = [self.cluster_labels.get(int(p), "Unknown") for p in preds]

        return _pd.DataFrame({"cluster_id": preds.tolist(), "cluster_label": labels})


# ── Fonction principale ────────────────────────────────────────────────────────

def train_model(config: dict | None = None) -> str:
    """
    Entraîner le pipeline TF-IDF + K-Means et logger dans MLflow local.

    Args:
        config: Dictionnaire de configuration (charge config.yml si None).

    Returns:
        run_id: ID du run MLflow.
    """
    if config is None:
        config = load_config()

    cfg_mlflow = config["mlflow"]
    cfg_tfidf  = config["tfidf"]
    cfg_kmeans = config["kmeans"]
    cfg_paths  = config["paths"]

    # ── 1. Configuration MLflow local ──────────────────────────────────
    mlflow.set_tracking_uri(cfg_mlflow["tracking_uri"])
    mlflow.set_experiment(cfg_mlflow["experiment_name"])
    print(f"[train] MLflow tracking URI : {cfg_mlflow['tracking_uri']}")
    print(f"[train] Expérience          : {cfg_mlflow['experiment_name']}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[train] Run ID : {run_id}")

        # ── 2. Ingestion ────────────────────────────────────────────────
        from steps.ingest import load_data
        print(f"\n[Step 1] Ingestion — {cfg_paths['train_data']}")
        df = load_data(cfg_paths["train_data"])
        mlflow.log_param("train_data_path", cfg_paths["train_data"])
        mlflow.log_param("n_samples", len(df))

        # ── 3. Nettoyage ────────────────────────────────────────────────
        from steps.clean import clean_and_extract_patterns
        print("\n[Step 2] Nettoyage & extraction de patterns")
        df = clean_and_extract_patterns(df)
        n_unique_patterns = df["log_pattern"].nunique()
        n_unique_messages = df["message_clean"].nunique()
        reduction_pct = (
            (1 - n_unique_patterns / n_unique_messages) * 100
            if n_unique_messages > 0 else 0
        )
        mlflow.log_metric("n_unique_patterns", n_unique_patterns)
        mlflow.log_metric("pattern_reduction_pct", round(reduction_pct, 1))

        # ── 4. Vectorisation TF-IDF ─────────────────────────────────────
        print("\n[Step 3] Vectorisation TF-IDF")
        max_features = cfg_tfidf["max_features"]
        min_df       = cfg_tfidf["min_df"]
        max_df       = cfg_tfidf["max_df"]

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            min_df=min_df,
            max_df=max_df,
            token_pattern=r"(?u)\b\w+\b",
        )
        X = vectorizer.fit_transform(df["log_pattern"])
        feature_names = vectorizer.get_feature_names_out()
        print(f"[train] Matrice TF-IDF : {X.shape[0]:,} docs × {X.shape[1]} features")

        mlflow.log_param("max_features", max_features)
        mlflow.log_param("min_df", min_df)
        mlflow.log_param("max_df", max_df)
        mlflow.log_metric("tfidf_vocabulary_size", len(feature_names))

        # ── 5. Évaluation k-range ───────────────────────────────────────
        print("\n[Step 4] Évaluation Elbow + Silhouette")
        n_init       = cfg_kmeans["n_init"]
        random_state = cfg_kmeans["random_state"]
        k_values     = [int(k.strip()) for k in cfg_kmeans["k_range"].split(",")]

        eval_results: dict = {"k": [], "inertia": [], "silhouette": []}
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            preds = km.fit_predict(X)
            sil = silhouette_score(X, preds)
            eval_results["k"].append(k)
            eval_results["inertia"].append(km.inertia_)
            eval_results["silhouette"].append(sil)
            print(f"  k={k:2d}: inertia={km.inertia_:10.0f}, silhouette={sil:.3f}")

        best_idx = int(np.argmax(eval_results["silhouette"]))
        best_k   = eval_results["k"][best_idx]
        print(f"  → Meilleur k : {best_k} (silhouette={eval_results['silhouette'][best_idx]:.3f})")
        mlflow.log_metric("best_k_silhouette", best_k)
        mlflow.log_metric("best_silhouette_score", round(eval_results["silhouette"][best_idx], 4))

        # ── 6. Entraînement K-Means final ───────────────────────────────
        n_clusters = cfg_kmeans["n_clusters"]
        print(f"\n[Step 5] Entraînement K-Means (k={n_clusters})")
        kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        )
        labels    = kmeans_model.fit_predict(X)
        sil_score = silhouette_score(X, labels)
        df["cluster_id"] = labels

        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("n_init", n_init)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("k_range", cfg_kmeans["k_range"])
        mlflow.log_metric("silhouette_score", round(sil_score, 4))
        mlflow.log_metric("inertia", round(kmeans_model.inertia_, 2))
        print(f"[train] Silhouette : {sil_score:.3f} | Inertia : {kmeans_model.inertia_:.0f}")

        # ── 7. Labeling des clusters ────────────────────────────────────
        print("\n[Step 6] Labeling des clusters")
        df, cluster_labels_map, top_words = label_clusters(df, kmeans_model, feature_names)
        for cid, label in cluster_labels_map.items():
            mlflow.log_param(f"cluster_{cid}_label", label)
            print(f"  Cluster {cid}: {label}")

        # Taux d'anomalies (logs dans clusters "Erreurs")
        error_mask   = df["cluster_label"].str.contains("Erreurs", case=False, na=False)
        anomaly_rate = float(error_mask.mean()) if len(df) > 0 else 0.0
        mlflow.log_metric("anomaly_rate", round(anomaly_rate, 6))
        print(f"[train] anomaly_rate : {anomaly_rate:.4f} ({anomaly_rate * 100:.2f}%)")

        # ── 8. Export résumé CSV ────────────────────────────────────────
        print("\n[Step 7] Export du résumé des clusters")
        rows = []
        for cid in range(kmeans_model.n_clusters):
            cluster_data = df[df["cluster_id"] == cid]
            rows.append({
                "ClusterID":       cid,
                "Label":           cluster_labels_map.get(cid, "Unknown"),
                "TopTerms":        ", ".join(top_words[cid][:8]),
                "ExampleLog":      cluster_data["message"].iloc[0] if len(cluster_data) > 0 else "",
                "Count":           int(len(cluster_data)),
                "Percentage":      round(len(cluster_data) / len(df) * 100, 1),
                "AvgResponseTime": round(cluster_data["response_time_ms"].mean(), 1),
            })
        summary = pd.DataFrame(rows).sort_values("Count", ascending=False)
        tmp_dir  = tempfile.mkdtemp()
        summary_path = os.path.join(tmp_dir, "log_clusters_summary.csv")
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, "outputs")
        print(f"[train] Résumé loggé dans MLflow : {summary_path}")

        # ── 9. Logger le modèle PyFunc dans MLflow + models/ ───────────
        print("\n[Step 8] Enregistrement du modèle")
        input_example  = pd.DataFrame({"message": ["level=error msg='connection refused'"]})
        output_example = pd.DataFrame({"cluster_id": [0], "cluster_label": ["Erreurs Serveur"]})
        signature      = infer_signature(input_example, output_example)

        artifact_path = cfg_mlflow["artifact_path"]
        model_name    = cfg_mlflow["model_name"]

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=LogClusteringPipelineModel(vectorizer, kmeans_model, cluster_labels_map),
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )
        print(f"[train] Modèle enregistré dans MLflow Registry : {model_name}")

        # Sauvegarde locale dans models/
        models_dir = cfg_paths["models_dir"]
        os.makedirs(models_dir, exist_ok=True)
        import pickle
        model_path = os.path.join(models_dir, "log_clustering_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "vectorizer":     vectorizer,
                "kmeans":         kmeans_model,
                "cluster_labels": cluster_labels_map,
                "feature_names":  feature_names.tolist(),
            }, f)
        print(f"[train] Modèle sauvegardé localement : {model_path}")

        # ── Résumé final ────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  [SUCCESS] ENTRAÎNEMENT TERMINÉ")
        print("=" * 60)
        print(f"  Logs traités      : {len(df):,}")
        print(f"  Clusters          : {n_clusters}")
        print(f"  Silhouette score  : {sil_score:.3f}")
        print(f"  Anomaly rate      : {anomaly_rate * 100:.2f}%")
        print(f"  MLflow run ID     : {run_id}")
        print(f"  Modèle local      : {model_path}")
        print("=" * 60)

        return run_id


if __name__ == "__main__":
    cfg = load_config()
    train_model(cfg)
