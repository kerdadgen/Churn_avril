# Utiliser l'image Python officielle
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Créer un répertoire pour les données de cache Streamlit
RUN mkdir -p ~/.streamlit

# Créer le fichier de configuration Streamlit
RUN echo "\
[server]\n\
port = 8501\n\
headless = true\n\
" > ~/.streamlit/config.toml

# Exposer le port 8501
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lancer Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
