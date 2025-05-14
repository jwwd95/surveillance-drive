import os
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

# === CONFIGURATION via Variables d'Environnement ===
RECIPIENT_EMAIL = os.environ.get("DEST_EMAIL")
EMAIL_SENDER = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("APP_PASSWORD")
EMAIL_USER = os.environ.get("EMAIL_USER", "jalfatimi@gmail.com")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")

# Constantes pour IMAP (Gmail)
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT = 993

# Fichiers YOLO
YOLO_WEIGHTS_FILE = "yolov3-tiny.weights"
YOLO_CFG_FILE = "yolov3-tiny.cfg"
COCO_NAMES_FILE = "coco.names"

# Variables globales
yolo_net = None
yolo_output_layers = None
yolo_classes = None

# === FONCTIONS UTILITAIRES ===
def log_message(message):
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {message}", flush=True)

# === INITIALISATION ===
def load_yolo_model():
    global yolo_net, yolo_output_layers, yolo_classes
    if not all(os.path.exists(f) for f in [YOLO_WEIGHTS_FILE, YOLO_CFG_FILE, COCO_NAMES_FILE]):
        log_message(f"ERREUR CRITIQUE: Un ou plusieurs fichiers YOLO sont manquants ({YOLO_WEIGHTS_FILE}, {YOLO_CFG_FILE}, {COCO_NAMES_FILE}).")
        return False
    try:
        yolo_net = cv2.dnn.readNet(YOLO_WEIGHTS_FILE, YOLO_CFG_FILE)
        layer_names = yolo_net.getLayerNames()
        unconnected_out_layers = yolo_net.getUnconnectedOutLayers()
        if isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 1:
            yolo_output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        elif isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 2:
            yolo_output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
        else:
            yolo_output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        with open(COCO_NAMES_FILE, 'r') as f:
            yolo_classes = [line.strip() for line in f.readlines()]
        if 'person' not in yolo_classes:
            log_message("ERREUR CRITIQUE: La classe 'person' n'est pas trouvée dans coco.names.")
            return False
        log_message("Modèle YOLO chargé avec succès.")
        return True
    except Exception as e:
        log_message(f"ERREUR CRITIQUE: Erreur lors du chargement du modèle YOLO : {e}")
        import traceback
        traceback.print_exc()
        return False

# === FONCTIONS DE TRAITEMENT ===
def detect_human(image_cv2):
    if image_cv2 is None:
        log_message("  Image non valide reçue pour la détection.")
        return None
    height, width = image_cv2.shape[:2]
    if height == 0 or width == 0:
        log_message("  Image vide reçue pour la détection.")
        return None
    blob = cv2.dnn.blobFromImage(image_cv2, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    try:
        outputs = yolo_net.forward(yolo_output_layers)
    except Exception as e:
        log_message(f"  Erreur pendant la propagation avant (forward pass) YOLO: {e}")
        return None
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id_index = np.argmax(scores)
            confidence = scores[class_id_index]
            if class_id_index < len(yolo_classes) and yolo_classes[class_id_index] == 'person' and confidence > 0.5:
                return True
    return False

def send_email_alert(recipient_email, image_bytes_for_attachment, image_name_for_email):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        log_message("  AVERTISSEMENT: EMAIL_SENDER ou APP_PASSWORD non configurés. Impossible d'envoyer l'email.")
        return
    msg = MIMEMultipart()
    msg['Subject'] = f'🛑 Humain détecté sur l’image: {image_name_for_email}'
    msg['From'] = EMAIL_SENDER
    msg['To'] = recipient_email
    body_text = f'Un humain a été détecté sur l’image "{image_name_for_email}" ci-jointe (reçue par email).'
    msg.attach(MIMEText
