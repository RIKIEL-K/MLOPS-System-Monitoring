"""
steps/predict.py — Chargement du modèle et prédiction

Charge le modèle PyFunc depuis MLflow Registry ou depuis le fichier
pickle local (models/log_clustering_model.pkl) et expose une fonction
`predict()` utilisée par app.py (FastAPI).
"""

import os
import pickle
from typing import Union

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml


def load_config(config_path: str = "config.yml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_from_registry(model_name: str, tracking_uri: str,
                              stage: str = "Production"):
    """
    Charger le modèle depuis le MLflow Model Registry local.

    Args:
        model_name:   Nom du modèle dans le Registry.
        tracking_uri: URI du tracking MLflow local.
        stage:        Stage souhaité ('Production', 'Staging', etc.)

    Returns:
        Modèle MLflow PyFunc chargé.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{stage}"
    print(f"[predict] Chargement depuis MLflow Registry : {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def load_model_from_pickle(pickle_path: str = "models/log_clustering_model.pkl") -> dict:
    """
    Charger le modèle depuis le fichier pickle local (fallback).

    Args:
        pickle_path: Chemin vers le fichier .pkl.

    Returns:
        Dict contenant vectorizer, kmeans, cluster_labels, feature_names.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not os.path.isfile(pickle_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {pickle_path}\n"
            "  → Lancez `python main.py` pour entraîner le modèle d'abord."
        )
    print(f"[predict] Chargement depuis pickle : {pickle_path}")
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def predict(messages: Union[list, pd.DataFrame],
            config_path: str = "config.yml") -> pd.DataFrame:
    """
    Prédire le cluster de logs à partir de messages texte.

    Essaie d'abord de charger via MLflow Registry, puis fallback sur pickle.

    Args:
        messages:    Liste de chaînes ou DataFrame avec colonne `message`.
        config_path: Chemin vers config.yml.

    Returns:
        DataFrame avec colonnes `cluster_id` et `cluster_label`.
    """
    config = load_config(config_path)

    # Normaliser l'input en DataFrame
    if isinstance(messages, list):
        input_df = pd.DataFrame({"message": messages})
    elif isinstance(messages, pd.DataFrame):
        input_df = messages
    else:
        input_df = pd.DataFrame({"message": [str(messages)]})

    # Essayer MLflow Registry en premier
    try:
        model = load_model_from_registry(
            model_name=config["mlflow"]["model_name"],
            tracking_uri=config["mlflow"]["tracking_uri"],
        )
        return model.predict(input_df)

    except Exception as mlflow_error:
        print(f"[predict] MLflow Registry indisponible ({mlflow_error}), "
              "fallback sur pickle local.")

    # Fallback : pickle local
    model_data = load_model_from_pickle(
        os.path.join(config["paths"]["models_dir"], "log_clustering_model.pkl")
    )

    from steps.clean import clean_message_for_inference

    vectorizer     = model_data["vectorizer"]
    kmeans         = model_data["kmeans"]
    cluster_labels = model_data["cluster_labels"]

    cleaned = [clean_message_for_inference(m) for m in input_df["message"].tolist()]
    X_pred  = vectorizer.transform(cleaned)
    preds   = kmeans.predict(X_pred)
    labels  = [cluster_labels.get(int(p), "Unknown") for p in preds]

    return pd.DataFrame({"cluster_id": preds.tolist(), "cluster_label": labels})


if __name__ == "__main__":
    # Test rapide
    test_messages = [
        "level=error msg='connection refused' service=api",
        "GET /health status=200 response_time=12ms",
        "POST /users/42 status=403 msg='forbidden'",
    ]
    results = predict(test_messages)
    print("\n=== Résultats de prédiction ===")
    for msg, (_, row) in zip(test_messages, results.iterrows()):
        print(f"  [{row['cluster_id']}] {row['cluster_label']} ← {msg[:60]}")
