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
APP_PASSWORD = os.environ.get("APP_PASSWORD")
DEST_EMAIL = os.environ.get("DEST_EMAIL", "jalfatimi@gmail.com")
PYTHONUNBUFFERED = os.environ.get("PYTHONUNBUFFERED", "1")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "saidben9560@gmail.com")

# Constantes pour IMAP (Gmail)
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT = 993
IMAP_TIMEOUT = 120

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
def preprocess_image(image_cv2):
    if image_cv2 is None:
        return None
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image_cv2.shape) == 3:
            lab = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            image_cv2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            image_cv2 = clahe.apply(image_cv2)
        image_cv2 = cv2.normalize(image_cv2, None, 0, 255, cv2.NORM_MINMAX)
        return image_cv2
    except Exception as e:
        log_message(f"  Erreur lors du prétraitement de l'image : {e}")
        return None

def detect_human(image_cv2, image_name_for_log):
    if image_cv2 is None:
        log_message(f"  Image non valide reçue pour la détection ({image_name_for_log}).")
        return None
    height, width = image_cv2.shape[:2]
    if height == 0 or width == 0:
        log_message(f"  Image vide reçue pour la détection ({image_name_for_log}, dimensions: {height}x{width}).")
        return None
    
    image_cv2 = preprocess_image(image_cv2)
    if image_cv2 is None:
        log_message(f"  Échec du prétraitement pour l'image ({image_name_for_log}).")
        return None

    blob = cv2.dnn.blobFromImage(image_cv2, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    try:
        outputs = yolo_net.forward(yolo_output_layers)
        log_message(f"  Détection YOLO terminée pour {image_name_for_log}, nombre de détections : {sum(len(output) for output in outputs)}")
    except Exception as e:
        log_message(f"  Erreur pendant la propagation avant (forward pass) YOLO pour {image_name_for_log}: {e}")
        return None
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id_index = np.argmax(scores)
            confidence = scores[class_id_index]
            if (class_id_index < len(yolo_classes) and 
                yolo_classes[class_id_index] == 'person' and 
                confidence > 0.25):
                log_message(f"  Détection : Humain trouvé dans {image_name_for_log} avec confiance {confidence:.2f}")
                return True
    log_message(f"  Aucun humain détecté dans {image_name_for_log}")
    return False

def send_email_alert(recipient_email, image_bytes_for_attachment, image_name_for_email, detection_time):
    if not SENDER_EMAIL or not APP_PASSWORD:
        log_message("  AVERTISSEMENT: SENDER_EMAIL ou APP_PASSWORD non configurés. Impossible d'envoyer l'email.")
        return
    msg = MIMEMultipart()
    subject = f"Humain présent à {detection_time.strftime('%H:%M')}"
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    body_text = f'Un humain a été détecté sur l’image "{image_name_for_email}" ci-jointe (reçue par email) à {detection_time.strftime("%H:%M")} UTC.'
    msg.attach(MIMEText(body_text, 'plain'))
    
    if image_bytes_for_attachment:
        try:
            log_message(f"  Taille des données de l'image {image_name_for_email}: {len(image_bytes_for_attachment)} octets")
            if image_name_for_email.lower().endswith(('.jpg', '.jpeg')):
                subtype = 'jpeg'
            elif image_name_for_email.lower().endswith('.png'):
                subtype = 'png'
            else:
                subtype = 'jpeg'
            img_mime = MIMEImage(image_bytes_for_attachment, _subtype=subtype)
            img_mime.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_name_for_email))
            msg.attach(img_mime)
            log_message(f"  Image {image_name_for_email} attachée avec succès à l'e-mail.")
        except Exception as e:
            log_message(f"  Erreur lors de l'attachement de l'image {image_name_for_email}: {e}")
            body_text += f"\n(Erreur lors de l'attachement de l'image : {str(e)})"
            msg.attach(MIMEText(body_text, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        log_message(f"  Email envoyé avec succès à {recipient_email} pour l'image {image_name_for_email}")
    except Exception as e:
        log_message(f"  Erreur lors de l'envoi de l'email : {e}")

# === SURVEILLANCE DES EMAILS ===
def connect_to_imap():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            mail = imaplib.IMAP4_SSL(SMTP_SERVER, SMTP_PORT)
            mail.socket().settimeout(IMAP_TIMEOUT)
            log_message(f"Tentative de connexion à imap.gmail.com (essai {attempt + 1}/{max_retries})...")
            mail.login(SENDER_EMAIL, APP_PASSWORD)
            log_message("Connexion IMAP réussie. Sélection de la boîte inbox...")
            mail.select("inbox")
            return mail
        except (imaplib.IMAP4.error, socket.timeout) as e:
            log_message(f"  Erreur lors de la connexion IMAP (essai {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                log_message("  Échec de la connexion IMAP après toutes les tentatives.")
                return None

def process_emails():
    log_message("Connexion à la boîte mail pour analyse...")
    mail = connect_to_imap()
    if not mail:
        log_message("  Échec de la connexion IMAP. Abandon.")
        return

    try:
        status, data = mail.search(None, "ALL")
        email_ids = data[0].split()
        log_message(f"Nombre d'emails trouvés : {len(email_ids)}")
        if not email_ids:
            log_message("  Aucun email trouvé dans la boîte.")
        
        failed_count = 0
        max_failures = 3
        for email_id in email_ids:
            try:
                mail.noop()
                log_message(f"Fetching email ID: {email_id.decode()}")
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                if status != 'OK' or not msg_data or len(msg_data) < 2:
                    log_message(f"  Échec de la récupération de l'email {email_id.decode()}: statut {status} ou données manquantes")
                    failed_count += 1
                    if failed_count >= max_failures:
                        log_message(f"  Trop d'échecs consécutifs ({failed_count}). Tentative de reconnexion...")
                        mail.logout()
                        mail = connect_to_imap()
                        if not mail:
                            log_message("  Échec de la reconnexion IMAP. Arrêt du traitement.")
                            return
                        failed_count = 0
                    continue
                failed_count = 0
                if not isinstance(msg_data[0], tuple) or len(msg_data[0]) < 2 or not isinstance(msg_data[0][1], bytes):
                    log_message(f"  Données invalides pour l'email {email_id.decode()} : type {type(msg_data[0][1]) if len(msg_data[0]) > 1 else 'None'}, valeur {msg_data[0][1] if len(msg_data[0]) > 1 else 'None'}")
                    continue
                raw_email = msg_data[0][1]
                if not isinstance(raw_email, bytes):
                    log_message(f"  Données invalides pour l'email {email_id.decode()} : type {type(raw_email)}, valeur {raw_email}")
                    continue
                msg = email.message_from_bytes(raw_email)
                if msg.is_multipart():
                    body_found = False
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            try:
                                charset = part.get_content_charset('utf-8')
                                body = part.get_payload(decode=True).decode(charset, errors='replace')
                                if ("Alarm event: Motion DetectStart" in body or 
                                    "Alarm event: Human DetectEnd" in body or 
                                    "Alarm event: Motion DetectEnd" in body or 
                                    "Alarm event: Human DetectStart" in body):
                                    log_message(f"  📧 Email {email_id.decode()} contient un mot-clé dans le corps.")
                                    body_found = True
                                    break
                            except (UnicodeDecodeError, AttributeError) as e:
                                log_message(f"  ⚠️ Erreur de décodage du corps de l'email {email_id.decode()}: {e}. Passage à l'attachement.")
                                continue
                    if body_found:
                        detection_time = datetime.datetime.now(datetime.timezone.utc)
                        if "DetectEnd" in body:
                            log_message(f"  📧 Email {email_id.decode()} contient 'DetectEnd'. Suppression immédiate.")
                            mail.store(email_id, '+FLAGS', '\\Deleted')
                            try:
                                mail.expunge()
                                log_message(f"  Suppression confirmée pour l'email {email_id.decode()} (DetectEnd).")
                            except Exception as e:
                                log_message(f"  Erreur lors de l'expunge pour l'email {email_id.decode()}: {e}")
                                mail.logout()
                                mail = connect_to_imap()
                                if mail:
                                    mail.store(email_id, '+FLAGS', '\\Deleted')
                                    mail.expunge()
                                    log_message(f"  Suppression confirmée après reconnexion pour l'email {email_id.decode()}.")
                            continue
                        has_attachment = False
                        for attachment_part in msg.walk():
                            if attachment_part.get_content_maintype() == 'multipart':
                                continue
                            if attachment_part.get('Content-Disposition') and 'attachment' in attachment_part.get('Content-Disposition'):
                                has_attachment = True
                                filename = attachment_part.get_filename()
                                if filename and (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
                                    log_message(f"  📧 Traitement de l'attachment {filename} dans l'email {email_id.decode()}...")
                                    image_data = attachment_part.get_payload(decode=True)
                                    log_message(f"  Taille des données de l'image {filename}: {len(image_data)} octets")
                                    try:
                                        with open("/tmp/tmp_image.jpg", "wb") as f:
                                            f.write(image_data)
                                        img_cv2 = cv2.imread("/tmp/tmp_image.jpg")
                                        if img_cv2 is None:
                                            raise ValueError("Image corrompue ou invalide")
                                    except Exception as e:
                                        log_message(f"  ⚠️ Image {filename} tronquée ou invalide : {e}")
                                        continue
                                    detection_result = detect_human(img_cv2, filename)
                                    if detection_result is True:
                                        log_message(f"  ✅ Humain détecté dans {filename}. Envoi de l’alerte email.")
                                        send_email_alert(DEST_EMAIL, image_data, filename, detection_time)
                                    elif detection_result is False:
                                        log_message(f"  ❌ Aucun humain détecté dans {filename}.")
                                    else:
                                        log_message(f"  ⚠️ Erreur de décodage/détection sur {filename}.")
                        mail.store(email_id, '+FLAGS', '\\Deleted')
                        try:
                            mail.expunge()
                            log_message(f"  Suppression confirmée pour l'email {email_id.decode()} après traitement.")
                        except Exception as e:
                            log_message(f"  Erreur lors de l'expunge pour l'email {email_id.decode()}: {e}")
                            mail.logout()
                            mail = connect_to_imap()
                            if mail:
                                mail.store(email_id, '+FLAGS', '\\Deleted')
                                mail.expunge()
                                log_message(f"  Suppression confirmée après reconnexion pour l'email {email_id.decode()}.")
                time.sleep(4)
            except (imaplib.IMAP4.error, socket.timeout, AttributeError) as e:
                log_message(f"  Erreur lors du traitement de l'email {email_id.decode()}: {e}")
                log_message("  Tentative de reconnexion IMAP...")
                mail.logout()
                mail = connect_to_imap()
                if not mail:
                    log_message("  Échec de la reconnexion IMAP. Arrêt du traitement.")
                    return
                continue
        
        mail.logout()
        log_message("Déconnexion IMAP réussie.")
        log_message("Analyse des emails terminée.")
    except Exception as e:
        log_message(f"  Erreur inattendue lors du traitement des emails : {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            mail.logout()
        except:
            pass

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

# === SCRIPT PRINCIPAL ===
def main():
    log_message("--- Initialisation du script de surveillance des emails ---")
    required_vars = {
        "SENDER_EMAIL": SENDER_EMAIL,
        "APP_PASSWORD": APP_PASSWORD,
        "DEST_EMAIL": DEST_EMAIL
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        log_message(f"ERREUR CRITIQUE: Variables d'environnement manquantes : {', '.join(missing_vars)}.")
        log_message("Veuillez vérifier leur configuration sur Koyeb.")
        return

    health_check_thread = threading.Thread(target=run_health_check_server, daemon=True)
    health_check_thread.start()
    log_message("Serveur de health check démarré.")

    if not load_yolo_model():
        log_message("Échec du chargement du modèle YOLO. Le script continuera mais la détection YOLO sera désactivée.")

    log_message(f"Emails analysés sur: {SENDER_EMAIL}")
    log_message(f"Emails envoyés à: {DEST_EMAIL}")
    log_message(f"Emails envoyés de: {SENDER_EMAIL}")
    log_message("----------------------------------------------------")

    log_message("Début de la boucle principale...")
    while True:
        try:
            log_message("--- Début d'une nouvelle exécution ---")
            process_emails()
            log_message("--- Fin de l'exécution, attente de 5 minutes ---")
            time.sleep(300)
        except Exception as e:
            log_message(f"Une erreur majeure est survenue dans main() : {e}")
            import traceback
            traceback.print_exc()
            time.sleep(300)

if __name__ == "__main__":
    main()
