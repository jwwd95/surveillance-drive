import os
import time
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
import io

# Configuration
FOLDER_NAME = "caméra_FR"
EMAIL_DEST = "JalFatimi@gmail.com"
GMAIL_USER = "saidben9560@gmail.com"  # Remplacez par votre adresse Gmail
GMAIL_PASSWORD = "ajut jinq dwkp pywj"  # Remplacez par un mot de passe d'application
CHECK_INTERVAL = 60  # Vérification toutes les 60 secondes

# Charger le modèle Tiny YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Authentification Google Drive
creds = Credentials.from_service_account_file("credentials.json", scopes=["https://www.googleapis.com/auth/drive.readonly"])
drive_service = build("drive", "v3", credentials=creds)

# Trouver l'ID du dossier
def get_folder_id(folder_name):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    folders = results.get("files", [])
    return folders[0]["id"] if folders else None

# Liste des fichiers déjà traités
processed_files = set()

def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = EMAIL_DEST
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        server.send_message(msg)

def detect_human(image_path):
    img = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 = humain dans YOLO
                return True
    return False

# Boucle principale
folder_id = get_folder_id(FOLDER_NAME)
if not folder_id:
    raise Exception("Dossier caméra_FR non trouvé")
while True:
    results = drive_service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()
    files = results.get("files", [])
    
    for file in files:
        file_id = file["id"]
        if file_id not in processed_files:
            # Télécharger l'image
            request = drive_service.files().get_media(fileId=file_id)
            fh = io.FileIO("temp_image.jpg", "wb")
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            # Analyser l'image
            if detect_human("temp_image.jpg"):
                send_email("Humain détecté", f"Un humain a été détecté dans {file['name']}.")
            processed_files.add(file_id)
            os.remove("temp_image.jpg")  # Nettoyer
    
    time.sleep(CHECK_INTERVAL)