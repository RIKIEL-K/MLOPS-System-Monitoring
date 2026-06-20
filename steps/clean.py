"""
steps/clean.py — Classe Cleaner

Nettoie les logs et extrait des patterns structurés pour la vectorisation TF-IDF.
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


class Cleaner:
    """
    Nettoie un DataFrame de logs et extrait la colonne `log_pattern`.

    Usage:
        cleaner = Cleaner()
        df_clean = cleaner.clean_data(df)
    """

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyer les messages et extraire des patterns opérationnels structurés.

        Chaque ligne de log est réduite à un pattern normalisé :
          "<method> <endpoint_normalisé> status_<code> <component> <action> <level>"

        Args:
            df: DataFrame de logs avec les colonnes requises.

        Returns:
            DataFrame enrichi avec `message_clean` et `log_pattern`.
        """
        df = df.copy()

        df["message_clean"] = (
            df["message"].astype(str).str.strip().str.lower()
        )
        df["log_pattern"] = df.apply(self._extract_pattern, axis=1)

        n_unique_patterns = df["log_pattern"].nunique()
        n_unique_messages = df["message_clean"].nunique()
        reduction = (
            (1 - n_unique_patterns / n_unique_messages) * 100
            if n_unique_messages > 0 else 0
        )
        logger.info(
            "Patterns extraits : %d uniques (réduction %.1f%% vs messages bruts)",
            n_unique_patterns, reduction,
        )
        return df

    # ── Private ────────────────────────────────────────────────────────

    def _extract_pattern(self, row: pd.Series) -> str:
        """Construire un pattern normalisé depuis une ligne de log."""
        parts = []

        parts.append(str(row.get("method", "unknown")).lower())

        endpoint = str(row.get("endpoint", "")).lower()
        endpoint = re.sub(r"/\d+", "/{id}", endpoint)
        endpoint_clean = endpoint.replace("/", "_").strip("_") or "root"
        parts.append(endpoint_clean)

        try:
            parts.append(f"status_{int(row['status_code'])}")
        except (ValueError, KeyError):
            parts.append("status_unknown")

        parts.append(str(row.get("component", "unknown")).lower())
        parts.append(str(row.get("action",    "unknown")).lower())
        parts.append(str(row.get("level",     "info")).lower())

        return " ".join(parts)

    @staticmethod
    def clean_message_for_inference(msg: str) -> str:
        """
        Normaliser un message brut pour l'inférence (API /predict).

        Remplace les tokens variables (nombres, UUIDs, IPs) par des
        placeholders pour que le modèle TF-IDF reconnaisse les patterns.

        Args:
            msg: Message de log brut.

        Returns:
            Message normalisé en minuscules.
        """
        if not isinstance(msg, str):
            return ""
        msg = re.sub(r"\b\d+\b", "<NUM>", msg)
        msg = re.sub(
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
            "<UUID>", msg, flags=re.IGNORECASE,
        )
        msg = re.sub(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "<IP>", msg)
        return msg.lower().strip()
