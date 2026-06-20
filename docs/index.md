# AIOps Log Clustering

Pipeline de clustering de logs (TF-IDF + K-Means) avec MLflow local.

## Démarrage Rapide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Générer le dataset
```bash
python dataset.py
```

### 3. Entraîner le modèle
```bash
python main.py
# ou : make train
```

### 4. Voir les résultats dans MLflow
```bash
mlflow ui
# Ouvrir http://localhost:5000
```

### 5. Lancer l'API de prédiction
```bash
uvicorn app:app --reload
# Ouvrir http://localhost:8000/docs
```

## Architecture

```
steps/
├── ingest.py    → Chargement et validation des données
├── clean.py     → Nettoyage et extraction de patterns
├── train.py     → TF-IDF + K-Means + MLflow local
└── predict.py   → Interface de prédiction pour l'API
```

## Documentation complète

```bash
mkdocs serve
# Ouvrir http://localhost:8001
```
