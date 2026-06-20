# AIOps Log Clustering

> Pipeline MLOps de clustering de logs avec TF-IDF, K-Means et MLflow local.

## Stack

| Composant | Outil |
|---|---|
| Versioning données | DVC |
| Tracking expériences | MLflow (local) |
| Serving | FastAPI |
| Orchestration | `main.py` |
| Tests | pytest |
| Documentation | MkDocs Material |

## Démarrage Rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Générer le dataset
python dataset.py

# 3. Entraîner le modèle
python main.py

# 4. Voir les runs MLflow
mlflow ui   # → http://localhost:5000

# 5. Servir l'API
uvicorn app:app --reload   # → http://localhost:8000/docs
```

## Architecture

```
monitoring-ia/
├── data/               ← Données (versionnées via DVC)
│   ├── train.csv
│   └── test.csv
├── models/             ← Modèles entraînés (gitignored)
├── mlruns/             ← Runs MLflow locaux (gitignored)
├── steps/              ← Pipeline ML
│   ├── ingest.py       ← Chargement et validation
│   ├── clean.py        ← Nettoyage et patterns
│   ├── train.py        ← TF-IDF + K-Means + MLflow
│   └── predict.py      ← Interface de prédiction
├── tests/              ← Tests unitaires
├── app.py              ← API FastAPI
├── main.py             ← Orchestrateur local
├── dataset.py          ← Génération train/test
└── config.yml          ← Configuration centralisée
```

## Commandes utiles

```bash
make train      # Entraîner
make test       # Tests
make serve      # Lancer l'API
make docs       # Documentation locale
```
