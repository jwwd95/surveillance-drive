# Utiliser une image Python 3.10 officielle, version slim pour la taille
FROM python:3.10.16-slim-bullseye

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires pour OpenCV et d'autres bibliothèques
# --no-install-recommends réduit la taille de l'image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Nettoyer le cache apt pour réduire la taille de l'image
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances Python
COPY requirements.txt .

# Installer les dépendances Python
# --no-cache-dir réduit la taille de l'image en n'enregistrant pas le cache pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers de l'application (script.py, .weights, .cfg)
# Assurez-vous que yolov3-tiny.weights et yolov3-tiny.cfg sont dans le même dossier que ce Dockerfile
COPY . .

# S'assurer que les logs Python sont affichés immédiatement (non bufferisés)
ENV PYTHONUNBUFFERED=1

# Commande pour exécuter l'application lorsque le conteneur démarre
CMD ["python", "script.py"]
