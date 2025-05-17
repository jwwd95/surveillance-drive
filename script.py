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
from PIL import Image
import io

# === CONFIGURATION via Variables d'Environnement ===
RECIPIENT_EMAIL = os.environ.get("DEST_EMAIL", "jalfatimi@gmail.com").lower()  # Normalisation de la casse
EMAIL_ADDRESS = os.environ.get("SENDER_EMAIL", "saidben9560@gmail.com")  # Utilise SENDER_EMAIL comme adresse principale
EMAIL_PASSWORD = os.environ.get("APP_PASSWORD")  # Utilise APP_PASSWORD comme mot de passe

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
        log_message("Modèle YOLO chargé avec succès. Classes disponibles: " + ", ".join(yolo_classes))
        return True
    except Exception as e:
        log_message(f"ERREUR CRITIQUE: Erreur lors du chargement du modèle YOLO : {e}")
        import traceback
        traceback.print_exc()
        return False

# === FONCTIONS DE TRAITEMENT ===
def preprocess_image(image_cv2, filename):
    if image_cv2 is None:
        log_message(f"  Échec du prétraitement pour {filename}: Image non valide ou non décodée.")
        return None
    try:
        # Vérification initiale des dimensions
        height, width = image_cv2.shape[:2]
        log_message(f"  Prétraitement de {filename}: Dimensions initiales {height}x{width}, type {image_cv2.dtype}, canaux {image_cv2.shape[-1] if len(image_cv2.shape) == 3 else 1}")
        
        # Redimensionnement
        target_size = (416, 416)
        image_cv2 = cv2.resize(image_cv2, target_size, interpolation=cv2.INTER_AREA)
        log_message(f"  Prétraitement: {filename} redimensionnée à {target_size}, nouvelle taille {image_cv2.shape}")

        # Réduction de bruit
        image_cv2 = cv2.GaussianBlur(image_cv2, (5, 5), 0)
        log_message(f"  Prétraitement: Réduction de bruit appliquée à {filename} avec flou gaussien (5x5)")

        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image_cv2.shape) == 3:
            lab = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            image_cv2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            image_cv2 = clahe.apply(image_cv2)
        log_message(f"  Prétraitement: Contraste amélioré pour {filename} avec CLAHE")

        # Normalisation
        image_cv2 = cv2.normalize(image_cv2, None, 0, 255, cv2.NORM_MINMAX)
        log_message(f"  Prétraitement: Normalisation appliquée à {filename}, valeurs min/max: {image_cv2.min()}/{image_cv2.max()}")
        
        return image_cv2
    except Exception as e:
        log_message(f"  ERREUR PRÉTRAITEMENT: {filename} a échoué - {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_human(image_cv2, image_name_for_log):
    if image_cv2 is None:
        log_message(f"  DÉTECTION: {image_name_for_log} non valide après prétraitement, abandon.")
        return None
    
    log_message(f"  DÉTECTION: Lancement de la détection sur {image_name_for_log}, taille {image_cv2.shape}, type {image_cv2.dtype}")
    blob = cv2.dnn.blobFromImage(image_cv2, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    try:
        outputs = yolo_net.forward(yolo_output_layers)
        log_message(f"  DÉTECTION: Propagation avant YOLO terminée pour {image_name_for_log}, nombre de détections: {len(outputs[0])}")
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id_index = np.argmax(scores)
                confidence = scores[class_id_index]
                log_message(f"  DÉTECTION: Analyse détection - classe {class_id_index}, confiance {confidence:.2f}")
                if class_id_index < len(yolo_classes) and yolo_classes[class_id_index] == 'person' and confidence > 0.3:
                    log_message(f"  ✅ DÉTECTION: Humain détecté dans {image_name_for_log} avec confiance {confidence:.2f}.")
                    return True
        log_message(f"  ⛔ DÉTECTION: Aucun humain détecté dans {image_name_for_log}.")
        return False
    except Exception as e:
        log_message(f"  ERREUR DÉTECTION: {image_name_for_log} a échoué - {e}")
        import traceback
        traceback.print_exc()
        return None

def send_email_alert(recipient_email, image_bytes_for_attachment, image_name_for_email):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        log_message("  AVERTISSEMENT: EMAIL_ADDRESS ou EMAIL_PASSWORD non configurés. Impossible d'envoyer l'email.")
        return
    msg = MIMEMultipart()
    msg['Subject'] = f'🛑 Humain détecté sur l’image: {image_name_for_email}'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = recipient_email
    body_text = f'Un humain a été détecté sur l’image "{image_name_for_email}" ci-jointe (reçue par email).'
    msg.attach(MIMEText(body_text, 'plain'))
    
    if image_bytes_for_attachment:
        try:
            log_message(f"  ENVOI EMAIL: Taille des données de {image_name_for_email}: {len(image_bytes_for_attachment)} octets")
            if image_name_for_email.lower().endswith(('.jpg', '.jpeg')):
                subtype = 'jpeg'
            elif image_name_for_email.lower().endswith('.png'):
                subtype = 'png'
            else:
                subtype = 'jpeg'
            log_message(f"  ENVOI EMAIL: Sous-type MIME pour {image_name_for_email}: {subtype}")
            img_mime = MIMEImage(image_bytes_for_attachment, _subtype=subtype)
            img_mime.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_name_for_email))
            msg.attach(img_mime)
            log_message(f"  ENVOI EMAIL: Image {image_name_for_email} attachée avec succès.")
        except Exception as e:
            log_message(f"  ERREUR ENVOI EMAIL: Attachement de {image_name_for_email} a échoué - {e}")
            body_text += f"\n(Erreur lors de l'attachement de l'image : {str(e)})"
            msg.attach(MIMEText(body_text, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        log_message(f"  ENVOI EMAIL: Email envoyé avec succès à {recipient_email} pour {image_name_for_email}")
    except Exception as e:
        log_message(f"  ERREUR ENVOI EMAIL: Échec de l'envoi à {recipient_email} pour {image_name_for_email} - {e}")

# === SURVEILLANCE DES EMAILS ===
def connect_to_imap():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            mail = imaplib.IMAP4_SSL(SMTP_SERVER, SMTP_PORT)
            mail.socket().settimeout(IMAP_TIMEOUT)
            log_message(f"IMAP: Tentative de connexion (essai {attempt + 1}/{max_retries})...")
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            log_message("IMAP: Connexion réussie. Sélection de la boîte inbox...")
            mail.select("inbox")
            return mail
        except (imaplib.IMAP4.error, socket.timeout) as e:
            log_message(f"IMAP: Échec de la connexion (essai {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                log_message("IMAP: Échec de la connexion après toutes les tentatives.")
                return None

def process_emails():
    log_message("IMAP: Connexion à la boîte mail pour analyse...")
    mail = connect_to_imap()
    if not mail:
        log_message("IMAP: Échec de la connexion IMAP. Abandon.")
        return

    try:
        since_date = (datetime.datetime.now() - datetime.timedelta(hours=6)).strftime("%d-%b-%Y")
        search_criteria = f'SINCE "{since_date}"'
        log_message(f"IMAP: Recherche des emails depuis {since_date} (lus et non lus)...")
        status, data = mail.search(None, search_criteria)
        email_ids = data[0].split()[:10]  # Limiter à 10 e-mails
        log_message(f"IMAP: {len(email_ids)} emails trouvés.")
        
        if not email_ids:
            log_message("IMAP: Aucun email trouvé dans la boîte.")
        
        for email_id in email_ids:
            try:
                log_message(f"IMAP: Récupération de l'email ID {email_id.decode()}...")
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                if status != 'OK' or not msg_data or len(msg_data) < 2 or msg_data[0] is None:
                    log_message(f"IMAP: Échec de la récupération de {email_id.decode()}: statut {status}, données {msg_data}")
                    continue
                raw_email = msg_data[0][1]
                log_message(f"IMAP: Données brutes de {email_id.decode()} récupérées, taille {len(raw_email)} octets")
                if not isinstance(raw_email, bytes):
                    try:
                        raw_email = str(raw_email).encode('utf-8')
                        log_message(f"IMAP: Conversion de {type(raw_email)} en bytes pour {email_id.decode()}.")
                    except Exception as e:
                        log_message(f"IMAP: Échec de conversion pour {email_id.decode()} - {e}")
                        continue
                msg = email.message_from_bytes(raw_email)
                if msg.is_multipart():
                    body_found = False
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            try:
                                charset = part.get_content_charset('utf-8')
                                body = part.get_payload(decode=True).decode(charset, errors='replace')
                                keywords = ["Alarm event: Motion DetectStart", "Alarm event: Human DetectEnd", "Alarm event: Motion DetectEnd"]
                                if any(keyword in body for keyword in keywords):
                                    log_message(f"  EMAIL: {email_id.decode()} contient un mot-clé dans le corps.")
                                    body_found = True
                                    break
                            except (UnicodeDecodeError, AttributeError) as e:
                                log_message(f"  EMAIL: Erreur de décodage du corps de {email_id.decode()} - {e}. Passage à l'attachement.")
                                continue
                    if body_found:
                        attachment_processed = False
                        for attachment_part in msg.walk():
                            if attachment_part.get_content_maintype() == 'multipart':
                                continue
                            if attachment_part.get('Content-Disposition') and 'attachment' in attachment_part.get('Content-Disposition'):
                                filename = attachment_part.get_filename()
                                if filename and (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
                                    log_message(f"  ATTACHEMENT: Traitement de {filename} dans {email_id.decode()}...")
                                    image_data = attachment_part.get_payload(decode=True)
                                    log_message(f"  ATTACHEMENT: Données de {filename} récupérées, taille {len(image_data)} octets, type {type(image_data)}")
                                    if image_data is None or len(image_data) == 0:
                                        log_message(f"  ATTACHEMENT: Données absentes ou vides pour {filename}.")
                                        continue
                                    log_message(f"  ATTACHEMENT: Premiers 20 octets de {filename}: {image_data[:20].hex()}")
                                    log_message(f"  ATTACHEMENT: Derniers 20 octets de {filename}: {image_data[-20:].hex() if len(image_data) > 20 else 'Insuffisant'}")
                                    # Décodage avec OpenCV
                                    img_cv2 = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                                    if img_cv2 is None:
                                        log_message(f"  DÉCODAGE: Échec avec OpenCV pour {filename}, tentative avec Pillow...")
                                        try:
                                            pil_image = Image.open(io.BytesIO(image_data))
                                            img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                                            log_message(f"  DÉCODAGE: Succès avec Pillow pour {filename}, dimensions {pil_image.size}")
                                        except Exception as e:
                                            log_message(f"  DÉCODAGE: Échec avec Pillow pour {filename} - {e}")
                                            img_cv2 = None
                                    # Prétraitement
                                    img_processed = preprocess_image(img_cv2, filename)
                                    if img_processed is None:
                                        log_message(f"  PRÉTRAITEMENT: Échec pour {filename}. L'email {email_id.decode()} est conservé.")
                                        continue
                                    # Détection
                                    detection_result = detect_human(img_processed, filename)
                                    if detection_result is not None:
                                        log_message(f"  DÉTECTION: Résultat pour {filename}: {detection_result}")
                                        if detection_result:
                                            log_message(f"  ACTION: Humain détecté dans {filename}. Envoi de l’alerte email à {RECIPIENT_EMAIL}.")
                                            send_email_alert(RECIPIENT_EMAIL, image_data, filename)
                                        log_message(f"  ACTION: Suppression de {email_id.decode()} après traitement réussi.")
                                        mail.store(email_id, '+FLAGS', '\\Deleted')
                                        try:
                                            mail.expunge()
                                            log_message(f"  ACTION: Suppression confirmée pour {email_id.decode()}.")
                                        except Exception as e:
                                            log_message(f"  ACTION: Erreur lors de l'expunge pour {email_id.decode()} - {e}")
                                    else:
                                        log_message(f"  ACTION: Échec de la détection pour {filename}. L'email {email_id.decode()} est conservé.")
                                    attachment_processed = True
                        if not attachment_processed:
                            log_message(f"  ACTION: Aucun attachement valide dans {email_id.decode()}. Conservé.")
                time.sleep(2)
            except (imaplib.IMAP4.error, socket.timeout, AttributeError) as e:
                log_message(f"  ERREUR: Traitement de {email_id.decode()} a échoué - {e}")
                log_message("  IMAP: Tentative de reconnexion...")
                mail.logout()
                mail = connect_to_imap()
                if not mail:
                    log_message("  IMAP: Échec de la reconnexion. Arrêt du traitement.")
                    return
                continue
        
        mail.logout()
        log_message("IMAP: Déconnexion réussie.")
        log_message("IMAP: Analyse des emails terminée.")
    except Exception as e:
        log_message(f"  ERREUR GLOBALE: Traitement des emails a échoué - {e}")
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
    log_message("HEALTH: Démarrage du serveur de health check sur le port 8000...")
    try:
        httpd.serve_forever()
    except Exception as e:
        log_message(f"HEALTH: Erreur dans le serveur de health check - {e}")

# === SCRIPT PRINCIPAL (pour boucle infinie) ===
def main():
    log_message("INIT: --- Initialisation du script de surveillance des emails ---")
    required_vars = {
        "DEST_EMAIL": RECIPIENT_EMAIL,
        "SENDER_EMAIL": EMAIL_ADDRESS,
        "APP_PASSWORD": EMAIL_PASSWORD
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        log_message(f"INIT: ERREUR CRITIQUE: Variables manquantes - {', '.join(missing_vars)}.")
        log_message("INIT: Vérifiez la configuration sur Koyeb.")
        return

    health_check_thread = threading.Thread(target=run_health_check_server, daemon=True)
    health_check_thread.start()
    log_message("INIT: Serveur de health check démarré.")

    if not load_yolo_model():
        log_message("INIT: Échec du chargement du modèle YOLO. La détection sera désactivée.")

    log_message(f"INIT: Emails analysés sur: {EMAIL_ADDRESS}")
    log_message(f"INIT: Emails envoyés à: {RECIPIENT_EMAIL}")
    log_message(f"INIT: Emails envoyés de: {EMAIL_ADDRESS}")
    log_message("INIT: ----------------------------------------------------")

    log_message("MAIN: Début de la boucle principale...")
    while True:
        try:
            log_message("MAIN: --- Début d'une nouvelle exécution ---")
            process_emails()
            log_message("MAIN: --- Fin de l'exécution, attente de 30 secondes ---")
            time.sleep(30)
        except Exception as e:
            log_message(f"MAIN: Erreur majeure - {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == "__main__":
    main()
