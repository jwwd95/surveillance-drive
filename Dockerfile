FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y libopencv-dev && apt-get clean

# Installer les dépendances Python
RUN pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client opencv-python numpy

# Copier les fichiers nécessaires
COPY script.py .
COPY surveillancedrive-93dee7913b77.json .
COPY yolov3-tiny.weights .
COPY yolov3-tiny.cfg .

# Exécuter le script
CMD ["python", "script.py"]