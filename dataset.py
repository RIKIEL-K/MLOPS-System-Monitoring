"""
dataset.py — Génère data/train.csv et data/test.csv

Lit le dataset brut (mock_loki_logs.csv), effectue un split 80/20,
et produit les deux fichiers cibles.

Les paramètres de split sont lus depuis params.yaml (DVC).

Usage:
    python dataset.py
    dvc repro prepare
"""

import argparse
import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str = "config.yml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_dataset(input_path: str, test_size: float, random_state: int,
                     train_out: str, test_out: str) -> None:
    if not os.path.isfile(input_path):
        print(f"[ERROR] Fichier source introuvable : {input_path}")
        sys.exit(1)

    print(f"[INFO] Lecture : {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] {len(df):,} lignes — colonnes : {list(df.columns)}")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Split → train: {len(train_df):,} | test: {len(test_df):,}")

    os.makedirs(os.path.dirname(train_out) or ".", exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out,  index=False)

    print(f"[OK] {train_out}")
    print(f"[OK] {test_out}")


def main() -> None:
    params = load_params()
    cfg    = load_config()

    parser = argparse.ArgumentParser(
        description="Génère data/train.csv et data/test.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",        type=str,   default=cfg["paths"]["raw_data"])
    parser.add_argument("--test-size",    type=float, default=params["split"]["test_size"])
    parser.add_argument("--random-state", type=int,   default=params["split"]["random_state"])
    parser.add_argument("--train-out",    type=str,   default=cfg["paths"]["train_data"])
    parser.add_argument("--test-out",     type=str,   default=cfg["paths"]["test_data"])
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
