"""
steps/ingest.py — Ingestion et validation des données de logs

Charge data/train.csv ou data/test.csv, valide le schéma attendu,
et retourne un DataFrame propre prêt pour le nettoyage.
"""

import os
import sys
from typing import Optional

import pandas as pd


# Colonnes minimales requises dans le CSV de logs
REQUIRED_COLUMNS = {
    "message",
    "level",
    "status_code",
    "method",
    "endpoint",
    "component",
    "action",
    "response_time_ms",
}

OPTIONAL_COLUMNS = {"timestamp", "logger", "message_raw"}


def load_data(csv_path: str, validate: bool = True) -> pd.DataFrame:
    """
    Charger le dataset CSV des logs et valider son schéma.

    Args:
        csv_path: Chemin vers le fichier CSV.
        validate: Si True, vérifie la présence des colonnes requises.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Dataset introuvable : {csv_path}\n"
            "  → Lancez d'abord `python dataset.py` pour générer train.csv et test.csv."
        )

    print(f"[ingest] Lecture de : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[ingest] {len(df):,} lignes chargées — {len(df.columns)} colonnes")

    if validate:
        _validate_schema(df, csv_path)

    # Forcer les types critiques
    df["status_code"] = pd.to_numeric(df["status_code"], errors="coerce").fillna(0).astype(int)
    df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce").fillna(0.0)

    # Supprimer les lignes sans message
    n_before = len(df)
    df = df.dropna(subset=["message"])
    if len(df) < n_before:
        print(f"[ingest] {n_before - len(df)} lignes supprimées (message vide)")

    print(f"[ingest] ✓ Dataset valide — {len(df):,} lignes utilisables")
    return df


def _validate_schema(df: pd.DataFrame, path: str) -> None:
    """Vérifier que toutes les colonnes requises sont présentes."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path} : {sorted(missing)}\n"
            f"  Colonnes présentes : {sorted(df.columns.tolist())}"
        )


def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Calculer des statistiques descriptives sur le dataset.

    Args:
        df: DataFrame de logs.

    Returns:
        Dict de statistiques : n_rows, level_dist, status_dist, avg_response_time.
    """
    stats = {
        "n_rows": len(df),
        "level_distribution": df["level"].value_counts().to_dict(),
        "status_code_distribution": df["status_code"].value_counts().head(10).to_dict(),
        "avg_response_time_ms": round(df["response_time_ms"].mean(), 2),
        "components": df["component"].unique().tolist() if "component" in df.columns else [],
    }
    return stats


if __name__ == "__main__":
    # Test rapide
    path = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    df = load_data(path)
    stats = get_data_stats(df)
    print("\n=== Statistiques ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
