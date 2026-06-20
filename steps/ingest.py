"""
steps/ingest.py — Classe Ingestion

Charge train.csv et test.csv, valide le schéma, et retourne deux DataFrames.
Lit les chemins depuis config.yml et les paramètres de split depuis params.yaml (DVC).
"""

import logging
import os

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "message", "level", "status_code", "method",
    "endpoint", "component", "action", "response_time_ms",
}


class Ingestion:
    """
    Charge les données depuis config.yml et les valide.

    Usage:
        ingestion = Ingestion()
        train, test = ingestion.load_data()
    """

    def __init__(self, config_path: str = "config.yml",
                 params_path: str = "params.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.train_path = cfg["paths"]["train_data"]
        self.test_path  = cfg["paths"]["test_data"]

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Charger et valider train.csv et test.csv.

        Returns:
            (train_df, test_df)

        Raises:
            FileNotFoundError: Si l'un des fichiers est manquant.
            ValueError: Si des colonnes requises sont absentes.
        """
        train = self._load_csv(self.train_path)
        test  = self._load_csv(self.test_path)
        logger.info("Train : %d lignes | Test : %d lignes", len(train), len(test))
        return train, test

    def _load_csv(self, path: str) -> pd.DataFrame:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Dataset introuvable : {path}\n"
                "  → Lancez `python dataset.py` ou `dvc repro prepare`."
            )

        df = pd.read_csv(path)
        self._validate(df, path)

        df["status_code"]      = pd.to_numeric(df["status_code"],      errors="coerce").fillna(0).astype(int)
        df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce").fillna(0.0)

        before = len(df)
        df = df.dropna(subset=["message"])
        if len(df) < before:
            logger.warning("%d ligne(s) supprimée(s) — message vide dans %s",
                           before - len(df), path)
        return df

    def _validate(self, df: pd.DataFrame, path: str) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {path} : {sorted(missing)}"
            )
