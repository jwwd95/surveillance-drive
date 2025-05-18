from flask import Flask
import threading
import smtplib
import cv2
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import datetime
import imaplib
import email
import os
import pytz

app = Flask(__name__)

# Configuration via variables d'environnement
APP_PASSWORD = os.environ.get("APP_PASSWORD")
DEST_EMAIL = os.environ.get("DEST_EMAIL", "jalfatimi@gmail.com")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "saidben9560@gmail.com")

# Variables globales pour YOLO
yolo_net = None
yolo_output_layers = None
yolo_classes = None

# Fonctions utilitaires (simplifiées pour l'exemple)
def log_message(message):
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {message}", flush=True)

def load_yolo_model():
    global yolo_net, yolo_output_layers, yolo_classes
    yolo_net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    with open("coco.names", 'r') as f:
        yolo_classes = [line.strip() for line in f.readlines()]
    yolo_output_layers = yolo_net.getLayerNames()
    log_message("Modèle YOLO chargé")
    return True

def detect_human_or_animal(image_cv2, image_name):
    # Logique de détection simplifiée
    log_message(f"Détection sur {image_name}")
    return "person", 0.9  # Exemple

def send_email_alert(image_name, detected_class):
    msg = MIMEMultipart()
    msg['Subject'] = f"{detected_class} détecté"
    msg['From'] = SENDER_EMAIL
    msg['To'] = DEST_EMAIL
    msg.attach(MIMEText(f"Détection sur {image_name}"))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
    log_message("Email envoyé")

# Route pour déclencher la surveillance
@app.route('/trigger')
def trigger_surveillance():
    log_message("Déclenchement manuel de la surveillance")
    # Simule la logique de process_emails
    mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
    mail.login(SENDER_EMAIL, APP_PASSWORD)
    mail.select("inbox")
    status, data = mail.search(None, "ALL")
    for email_id in data[0].split():
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "Motion DetectStart" in str(part.get_payload()):
                for att in msg.walk():
                    if att.get_content_type().startswith("image/"):
                        image_data = att.get_payload(decode=True)
                        with open("temp.jpg", "wb") as f:
                            f.write(image_data)
                        img = cv2.imread("temp.jpg")
                        detected_class, confidence = detect_human_or_animal(img, "temp.jpg")
                        if detected_class:
                            send_email_alert("temp.jpg", detected_class)
    mail.logout()
    return "Surveillance déclenchée"

# Boucle continue en arrière-plan
def run_background():
    while True:
        log_message("Vérification en cours...")
        time.sleep(300)  # Vérification toutes les 5 minutes si pas déclenché

if __name__ == "__main__":
    if load_yolo_model():
        threading.Thread(target=run_background, daemon=True).start()
        app.run(host="0.0.0.0", port=8080)
