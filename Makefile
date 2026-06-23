
PYTHON  := python
PIP     := pip
UVICORN := uvicorn
CONFIG  := config.yml

.PHONY: setup dataset train serve mlflow-ui test docs \
        dvc-init dvc-repro dvc-exp dvc-metrics dvc-plots lint clean help

help:
	@echo ""
	@echo "  Commandes disponibles"
	@echo "  ---------------------------------------------"
	@echo "  make setup        Installer les dependances Python"
	@echo "  make dataset      Generer data/train.csv et data/test.csv"
	@echo "  make train        Entrainer le modele (MLflow local)"
	@echo "  make serve        Lancer l'API FastAPI (port 8000)"
	@echo "  make mlflow-ui    Ouvrir l'interface MLflow (port 5000)"
	@echo "  make docs         Lancer la documentation MkDocs (port 8001)"
	@echo ""
	@echo "  -- DVC -------------------------------------------"
	@echo "  make dvc-init     Initialiser DVC + tracker les donnees"
	@echo "  make dvc-repro    Relancer le pipeline (si changements)"
	@echo "  make dvc-exp      Lancer une experience DVC (ex: n_clusters=8)"
	@echo "  --------------------------------------------------"
	@echo "  make lint         Verifier le style du code"
	@echo "  make clean        Nettoyer les fichiers temporaires"
	@echo ""

setup:
	$(PIP) install -r requirements.txt

dataset:
	$(PYTHON) dataset.py

train:
	$(PYTHON) main.py --config $(CONFIG)

serve:
	$(UVICORN) app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	$(PYTHON) -m webbrowser http://127.0.0.1:5000
	mlflow ui --backend-store-uri ./mlruns --port 5000

docs:
	mkdocs serve --dev-addr 0.0.0.0:8001

dvc-init:
	dvc init
	dvc add data/mock_loki_logs.csv
	@echo "DVC initialisé et données trackées."
	@echo "Configurez un remote : dvc remote add -d myremote <path>"

dvc-repro:
	dvc repro

dvc-exp:
	@echo "Exemple : dvc exp run -S kmeans.n_clusters=8"
	@echo "Exemple : dvc exp run -S tfidf.max_features=150 -S kmeans.n_clusters=8"
	dvc exp run
	
dvc-plots:
	dvc plots show plots/cluster_distribution.csv
