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
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# === CONFIGURATION via Variables d'Environnement ===
RECIPIENT_EMAIL = os.environ.get("DEST_EMAIL")
EMAIL_SENDER = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("APP_PASSWORD")
EMAIL_USER = os.environ.get("EMAIL_USER", "jalfatimi@gmail.com")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")

# Constantes pour IMAP (Gmail)
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT = 993
IMAP_TIMEOUT = 30  # Timeout de 30 secondes pour la connexion

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
    msg.attach(MIMEText(body_text, 'plain'))
    if image_bytes_for_attachment:
        try:
            subtype = 'jpeg' if image_name_for_email.lower().endswith(('.jpg', '.jpeg')) else 'png'
            img_mime = MIMEImage(image_bytes_for_attachment, subtype=subtype, name=os.path.basename(image_name_for_email))
            img_mime.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_name_for_email))
            msg.attach(img_mime)
        except Exception as e:
            log_message(f"  Erreur lors de l'attachement de l'image {image_name_for_email}: {e}")
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        log_message(f"  Email envoyé avec succès à {recipient_email} pour l'image {image_name_for_email}")
    except Exception as e:
        log_message(f"  Erreur lors de l'envoi de l'email : {e}")

# === SURVEILLANCE DES EMAILS ===
def process_emails():
    log_message("Connexion à la boîte mail pour analyse...")
    try:
        mail = imaplib.IMAP4_SSL(SMTP_SERVER, SMTP_PORT)
        mail.socket().settimeout(IMAP_TIMEOUT)
        log_message("Tentative de connexion à imap.gmail.com...")
        mail.login(EMAIL_USER, EMAIL_APP_PASSWORD)
        log_message("Connexion IMAP réussie. Sélection de la boîte inbox...")
        mail.select("inbox")
        # Limiter aux e-mails des dernières 24 heures
        since_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%d-%b-%Y")
        status, data = mail.search(None, f'SINCE "{since_date}"')
        email_ids = data[0].split()
        log_message(f"Nombre d'emails trouvés (depuis {since_date}) : {len(email_ids)}")
        if not email_ids:
            log_message("  Aucun email trouvé dans la boîte.")
        for email_id in email_ids:
            log_message(f"Fetching email ID: {email_id.decode()}")
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            if msg.is_multipart():
                body_found = False
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        try:
                            charset = part.get_content_charset('utf-8')
                            body = part.get_payload(decode=True).decode(charset, errors='replace')
                            if "Alarm event: Motion DetectStart" in body or "Alarm event: Human DetectEnd" in body:
                                log_message(f"  📧 Email {email_id.decode()} contient un mot-clé dans le corps.")
                                body_found = True
                                break
                        except (UnicodeDecodeError, AttributeError) as e:
                            log_message(f"  ⚠️ Erreur de décodage du corps de l'email {email_id.decode()}: {e}. Passage à l'attachement.")
                            continue
                if body_found:
                    for attachment_part in msg.walk():
                        if attachment_part.get_content_maintype() == 'multipart':
                            continue
                        if attachment_part.get('Content-Disposition') and 'attachment' in attachment_part.get('Content-Disposition'):
                            filename = attachment_part.get_filename()
                            if filename and (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
                                log_message(f"  📧 Traitement de l'attachment {filename} dans l'email {email_id.decode()}...")
                                image_data = attachment_part.get_payload(decode=True)
                                img_cv2 = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                                detection_result = detect_human(img_cv2)
                                if detection_result is True:
                                    log_message(f"  ✅ Humain détecté dans {filename}. Envoi de l’alerte email.")
                                    send_email_alert(EMAIL_USER, image_data, filename)
                                elif detection_result is False:
                                    log_message(f"  ❌ Aucun humain détecté dans {filename}. Suppression de l'email et de l'attachment.")
                                    mail.store(email_id, '+FLAGS', '\\Deleted')
                                else:
                                    log_message(f"  ⚠️ Erreur de décodage/détection sur {filename}. Non traité.")
        mail.expunge()
        log_message("Expunge terminé.")
        mail.logout()
        log_message("Déconnexion IMAP réussie.")
        log_message("Analyse des emails terminée.")
    except (imaplib.IMAP4.error, socket.timeout) as e:
        log_message(f"  Erreur lors de la connexion ou du traitement des emails : {e}")
    except Exception as e:
        log_message(f"  Erreur inattendue lors du traitement des emails : {e}")
        import traceback
        traceback.print_exc()

# === HEALTH CHECK SERVER ===
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

def run_health_check_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, HealthCheckHandler)
    log_message("Démarrage du serveur de health check sur le port 8000...")
    try:
        httpd.serve_forever()
    except Exception as e:
        log_message(f"Erreur dans le serveur de health check : {e}")

# === SCRIPT PRINCIPAL (pour boucle infinie) ===
def main():
    log_message("--- Initialisation du script de surveillance des emails ---")
    required_vars = {
        "SENDER_EMAIL": EMAIL_SENDER,
        "APP_PASSWORD": EMAIL_PASSWORD,
        "DEST_EMAIL": RECIPIENT_EMAIL,
        "EMAIL_APP_PASSWORD": EMAIL_APP_PASSWORD
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        log_message(f"ERREUR CRITIQUE: Variables d'environnement manquantes : {', '.join(missing_vars)}.")
        log_message("Veuillez vérifier leur configuration sur Koyeb.")
        return

    if not load_yolo_model():
        log_message("Échec du chargement du modèle YOLO. Arrêt.")
        return

    log_message(f"Emails analysés sur: {EMAIL_USER}")
    log_message(f"Emails envoyés à: {RECIPIENT_EMAIL}")
    log_message(f"Emails envoyés de: {EMAIL_SENDER}")
    log_message("----------------------------------------------------")

    # Démarrer le serveur de health check dans un thread séparé
    health_check_thread = threading.Thread(target=run_health_check_server, daemon=True)
    health_check_thread.start()
    log_message("Serveur de health check démarré. Début de la boucle principale...")

    while True:
        try:
            log_message("--- Début d'une nouvelle exécution ---")
            process_emails()
            log_message("--- Fin de l'exécution, attente de 5 minutes ---")
            time.sleep(300)  # Attendre 5 minutes avant la prochaine exécution
        except Exception as e:
            log_message(f"Une erreur majeure est survenue dans main() : {e}")
            import traceback
            traceback.print_exc()
            time.sleep(300)  # En cas d'erreur, attendre avant de réessayer

if __name__ == "__main__":
    main()
