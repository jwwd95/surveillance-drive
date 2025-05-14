FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système (OpenCV)
RUN apt-get update && apt-get install -y libopencv-dev && apt-get clean

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les autres fichiers nécessaires
COPY script.py .
COPY surveillancedrive-93dee7913b77.json .
COPY yolov3-tiny.weights .
COPY yolov3-tiny.cfg .

# Lancer le script
CMD ["python", "script.py"]
