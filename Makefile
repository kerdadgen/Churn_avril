.PHONY: help install test train run-api run-streamlit build docker-build docker-run clean

help:
	@echo "Commandes disponibles:"
	@echo "  make install        - Installe les dépendances"
	@echo "  make test           - Exécute les tests"
	@echo "  make train          - Entraîne le modèle"
	@echo "  make run-api        - Lance l'API Flask"
	@echo "  make run-streamlit  - Lance l'app Streamlit"
	@echo "  make docker-build   - Construit l'image Docker"
	@echo "  make docker-run     - Lance le container Docker"
	@echo "  make clean          - Nettoie les fichiers temporaires"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test: install
	pytest tests/ -v --tb=short

train: install
	python train.py

run-api: install
	python app.py

run-streamlit: install
	streamlit run streamlit_app.py

docker-build:
	docker build -t churn_avril .

docker-run: docker-build
	docker run -p 8501:8501 churn_avril

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.streamlit' -exec rm -rf {} + 2>/dev/null || true
