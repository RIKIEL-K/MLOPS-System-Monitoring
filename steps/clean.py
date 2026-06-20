"""
steps/clean.py — Nettoyage des logs et extraction de patterns

Transforme un DataFrame brut de logs en un DataFrame enrichi avec
la colonne `log_pattern` prête pour la vectorisation TF-IDF.
"""

import re
import pandas as pd


def clean_and_extract_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyer les messages et extraire des patterns opérationnels structurés.

    Chaque ligne de log est réduite à un pattern normalisé :
      "<method> <endpoint_normalisé> status_<code> <component> <action> <level>"

    Cette normalisation aide TF-IDF à généraliser sur les patterns
    plutôt que sur les valeurs concrètes (IDs, IPs, etc.).

    Args:
        df: DataFrame de logs avec les colonnes requises.

    Returns:
        DataFrame enrichi avec les colonnes `message_clean` et `log_pattern`.
    """
    df = df.copy()

    # Nettoyage basique du message texte
    df["message_clean"] = (
        df["message"].astype(str)
        .str.strip()
        .str.lower()
    )

    # Extraction du pattern structuré
    df["log_pattern"] = df.apply(_extract_pattern, axis=1)

    n_unique_patterns = df["log_pattern"].nunique()
    n_unique_messages = df["message_clean"].nunique()
    reduction = (1 - n_unique_patterns / n_unique_messages) * 100 if n_unique_messages > 0 else 0

    print(f"[clean] Patterns extraits : {n_unique_patterns} uniques "
          f"(réduction {reduction:.1f}% vs messages bruts)")

    return df


def _extract_pattern(row: pd.Series) -> str:
    """
    Construire un pattern normalisé depuis une ligne de log.

    Remplace les IDs numériques dans les endpoints par `{id}`
    pour éviter la sur-spécificité du modèle.
    """
    parts = []

    # Méthode HTTP
    parts.append(str(row.get("method", "unknown")).lower())

    # Endpoint normalisé : /users/42 → /users/{id}
    endpoint = str(row.get("endpoint", "")).lower()
    endpoint = re.sub(r"/\d+", "/{id}", endpoint)
    endpoint_clean = endpoint.replace("/", "_").strip("_") or "root"
    parts.append(endpoint_clean)

    # Code de statut HTTP
    try:
        parts.append(f"status_{int(row['status_code'])}")
    except (ValueError, KeyError):
        parts.append("status_unknown")

    # Composant applicatif
    parts.append(str(row.get("component", "unknown")).lower())

    # Action métier
    parts.append(str(row.get("action", "unknown")).lower())

    # Niveau de log
    parts.append(str(row.get("level", "info")).lower())

    return " ".join(parts)


def clean_message_for_inference(msg: str) -> str:
    """
    Nettoyer un message brut pour l'inférence (API /predict).

    Remplace les tokens variables (nombres, UUIDs, IPs) par des
    placeholders pour que le modèle TF-IDF reconnaisse les patterns.
    """
    if not isinstance(msg, str):
        return ""

    # Remplacer les nombres isolés
    msg = re.sub(r"\b\d+\b", "<NUM>", msg)

    # Remplacer les UUIDs
    msg = re.sub(
        r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
        "<UUID>",
        msg,
        flags=re.IGNORECASE,
    )

    # Remplacer les adresses IP
    msg = re.sub(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "<IP>", msg)

    return msg.lower().strip()


if __name__ == "__main__":
    # Test rapide
    import sys
    from steps.ingest import load_data

    path = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    df = load_data(path)
    df_clean = clean_and_extract_patterns(df)
    print(df_clean[["message", "log_pattern"]].head(5).to_string())
