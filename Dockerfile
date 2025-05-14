FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y libopencv-dev && apt-get clean

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers nécessaires
COPY . .

# Démarrer le script
ENTRYPOINT ["python", "script.py"]
