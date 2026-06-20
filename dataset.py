"""
dataset.py — Génère data/train.csv et data/test.csv

Lit le dataset brut de logs (mock_loki_logs.csv ou log_dataset.csv),
effectue un split 80/20, et produit les deux fichiers cibles.

Usage:
    python dataset.py
    python dataset.py --input data/mock_loki_logs.csv --test-size 0.2
"""

import argparse
import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config.yml") -> dict:
    """Charger la configuration depuis config.yml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_dataset(input_path: str, test_size: float, random_state: int,
                     train_out: str, test_out: str) -> None:
    """
    Lire le CSV source, splitter en train/test, sauvegarder.

    Args:
        input_path:   Chemin du CSV source.
        test_size:    Fraction réservée au test (0.0–1.0).
        random_state: Seed pour la reproductibilité.
        train_out:    Chemin de sortie pour train.csv.
        test_out:     Chemin de sortie pour test.csv.
    """
    if not os.path.isfile(input_path):
        print(f"[ERROR] Fichier source introuvable : {input_path}")
        sys.exit(1)

    print(f"[INFO] Lecture du dataset source : {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] {len(df):,} lignes chargées — colonnes : {list(df.columns)}")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Split → train: {len(train_df):,} lignes | test: {len(test_df):,} lignes")

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"[OK] train.csv sauvegardé : {train_out}")
    print(f"[OK] test.csv  sauvegardé : {test_out}")


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Génère data/train.csv et data/test.csv depuis un fichier source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/mock_loki_logs.csv",
        help="Chemin du fichier CSV source",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=cfg["split"]["test_size"],
        help="Fraction du dataset réservée au test",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=cfg["split"]["random_state"],
        help="Seed pour la reproductibilité",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default=cfg["paths"]["train_data"],
        help="Chemin de sortie pour train.csv",
    )
    parser.add_argument(
        "--test-out",
        type=str,
        default=cfg["paths"]["test_data"],
        help="Chemin de sortie pour test.csv",
    )
    args = parser.parse_args()

    generate_dataset(
        input_path=args.input,
        test_size=args.test_size,
        random_state=args.random_state,
        train_out=args.train_out,
        test_out=args.test_out,
    )


if __name__ == "__main__":
    main()
