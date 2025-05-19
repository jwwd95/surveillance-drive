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
import requests

app = Flask(__name__)

# Configuration via variables d'environnement
APP_PASSWORD = os.environ.get("APP_PASSWORD")
DEST_EMAIL = os.environ.get("DEST_EMAIL", "jalfatimi@gmail.com")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "saidben9560@gmail.com")
KOYEB_API_TOKEN = os.environ.get("KOYEB_API_TOKEN", "tffsuh11ifyybt2mjqaam1xz5z2ahe88tx8yqsidy0p40jihz6eqe9c06ieumuzt")
KOYEB_SERVICE_ID = os.environ.get("KOYEB_SERVICE_ID", "dbf9b5e8-b828-4a5b-853e-21bf0cf1fa10")
KOYEB_DEPLOYMENT_ID = os.environ.get("KOYEB_DEPLOYMENT_ID", "f9c47b30-2524-4132-9439-37fa1f39d979")  # Remplace par l'ID de déploiement actif

# Variables globales pour YOLO
yolo_net = None
yolo_output_layers = None
yolo_classes = None

# Fonctions utilitaires
def log_message(message):
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {message}", flush=True)

def load_yolo_model():
    global yolo_net, yolo_output_layers, yolo_classes
    try:
        yolo_net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
        layer_names = yolo_net.getLayerNames()
        unconnected_out_layers = yolo_net.getUnconnectedOutLayers()
        yolo_output_layers = [layer_names[i - 1] for i in unconnected_out_layers] if unconnected_out_layers.ndim == 1 else [layer_names[i[0] - 1] for i in unconnected_out_layers]
        with open("coco.names", 'r') as f:
            yolo_classes = [line.strip() for line in f.readlines()]
        log_message("Modèle YOLO chargé")
        return True
    except Exception as e:
        log_message(f"Erreur chargement YOLO : {e}")
        return False

def detect_human_or_animal(image_cv2, image_name):
    if image_cv2 is None or not image_cv2.shape[:2][0] or not image_cv2.shape[:2][1]: return None, None
    blob = cv2.dnn.blobFromImage(image_cv2, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    try:
        outputs = yolo_net.forward(yolo_output_layers)
        for output in outputs:
            for detection in output:
                class_id = np.argmax(detection[5:])
                confidence = detection[5:][class_id]
                if class_id < len(yolo_classes) and confidence > 0.25 and yolo_classes[class_id] in ["person", "cat"]:
                    log_message(f"Détection : {yolo_classes[class_id]} dans {image_name} ({confidence:.2f})")
                    return yolo_classes[class_id], confidence
    except Exception as e:
        log_message(f"Erreur YOLO pour {image_name}: {e}")
    return None, None

def send_email_alert(image_name, image_data, detected_class):
    msg = MIMEMultipart()
    msg['Subject'] = f"{detected_class} détecté"
    msg['From'] = SENDER_EMAIL
    msg['To'] = DEST_EMAIL
    msg.attach(MIMEText(f"Détection sur {image_name}"))
    if image_data:
        msg.attach(MIMEImage(image_data, _subtype="jpeg"))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
    log_message("Email envoyé")

# Route pour health check
@app.route('/')
@app.route('/health')
def health_check():
    log_message("Health check réussi")
    return "OK", 200

# Route pour déclencher la surveillance
@app.route('/trigger')
def trigger_surveillance():
    log_message("Déclenchement manuel de la surveillance")
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
                            send_email_alert("temp.jpg", image_data, detected_class)
                        mail.store(email_id, '+FLAGS', '\\Deleted')
                        mail.expunge()
    mail.logout()
    return "Surveillance déclenchée"

# Route pour redémarrer le service via l'API de Koyeb
@app.route('/restart')
def restart_service():
    log_message("Appel à /restart reçu")
    if not KOYEB_API_TOKEN or not KOYEB_SERVICE_ID or not KOYEB_DEPLOYMENT_ID:
        log_message("Erreur : KOYEB_API_TOKEN, KOYEB_SERVICE_ID ou KOYEB_DEPLOYMENT_ID manquant")
        return "Erreur de configuration", 500
    headers = {"Authorization": f"Bearer {KOYEB_API_TOKEN}", "Content-Type": "application/json"}
    base_url = "https://app.koyeb.com/v1"
    # Vérifier l'état actuel
    try:
        status_response = requests.get(f"{base_url}/services/{KOYEB_SERVICE_ID}", headers=headers, timeout=5)
        if status_response.status_code != 200:
            log_message(f"Erreur API état : {status_response.text}")
            return f"Erreur API état: {status_response.text}", 500
        log_message(f"Réponse API état : {status_response.text}")
        service_data = status_response.json()
        service_status = service_data.get("service", {}).get("status")
        log_message(f"État actuel : {service_status}")
        if service_status is None:
            log_message("État non récupéré, arrêt du processus")
            return "État non récupéré", 500
        if service_status not in ["HEALTHY", "STARTING"]:
            log_message(f"Service dans un état non redéployable : {service_status}")
            return f"Service dans un état non redéployable : {service_status}", 400
    except Exception as e:
        log_message(f"Exception vérification état : {str(e)}")
        return f"Exception vérification: {str(e)}", 500
    # Lancer le redéploiement
    log_message("Tentative de redéploiement...")
    try:
        response = requests.post(f"{base_url}/deployments/{KOYEB_DEPLOYMENT_ID}/redeploy", headers=headers, timeout=5)
        if response.status_code != 200:
            log_message(f"Erreur redéploiement : {response.text}")
            return f"Erreur redéploiement: {response.text}", 500
        log_message("Redéploiement initié")
    except Exception as e:
        log_message(f"Exception redéploiement : {str(e)}")
        return f"Exception redéploiement: {str(e)}", 500
    log_message("Redémarrage terminé")
    return "Service redémarré", 200

# Boucle en arrière-plan
def run_background():
    while True:
        log_message("Vérification en arrière-plan...")
        time.sleep(300)

if __name__ == "__main__":
    if load_yolo_model():
        threading.Thread(target=run_background, daemon=True).start()
        app.run(host="0.0.0.0", port=8000)
