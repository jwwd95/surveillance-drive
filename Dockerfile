# Utiliser une image Python 3.10 slim (ou la version que vous préférez)
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires pour OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances Python
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les autres fichiers du projet (script, fichiers yolo, etc.)
COPY . .

# Commande par défaut pour exécuter le script quand le conteneur démarre
# Koyeb l'utilisera pour le cron job. Le script est conçu pour s'exécuter une fois et se terminer.
CMD ["python", "script.py"]
