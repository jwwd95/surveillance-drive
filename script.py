import os
import io
import smtplib
import mimetypes
import cv2
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage # Pour attachement direct
import time
import base64
import json
import datetime

# === CONFIGURATION via Variables d'Environnement (Noms alignés sur votre config Koyeb) ===
GDRIVE_FOLDER_ID = os.environ.get("FOLDER_ID") # Votre nom: FOLDER_ID
RECIPIENT_EMAIL = os.environ.get("DEST_EMAIL") # Votre nom: DEST_EMAIL
EMAIL_SENDER = os.environ.get("SENDER_EMAIL") # Votre nom: SENDER_EMAIL
EMAIL_PASSWORD = os.environ.get("APP_PASSWORD") # Votre nom: APP_PASSWORD
SCOPES = ['https://www.googleapis.com/auth/drive']

GOOGLE_CREDENTIALS_BASE64 = os.environ.get('GOOGLE_CREDENTIALS_BASE64')

# Comportement de suppression (vous pouvez ajouter cette variable sur Koyeb si besoin)
DELETE_PROCESSED_FILES = os.environ.get('DELETE_PROCESSED_FILES', 'true').lower() == 'true'

# Fichiers YOLO
YOLO_WEIGHTS_FILE = "yolov3-tiny.weights"
YOLO_CFG_FILE = "yolov3-tiny.cfg"
COCO_NAMES_FILE = "coco.names"

PROCESSED_LOG_FILE = "processed_log.txt"

# Variables globales pour le service Drive et le modèle YOLO
gdrive_service = None
yolo_net = None
yolo_output_layers = None
yolo_classes = None

# === FONCTIONS UTILITAIRES ===
def log_message(message):
    print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {message}", flush=True)

# === INITIALISATION ===
def initialize_drive_api():
    global gdrive_service
    if not GOOGLE_CREDENTIALS_BASE64:
        log_message("ERREUR CRITIQUE: La variable d'environnement GOOGLE_CREDENTIALS_BASE64 n'est pas définie.")
        return False
    try:
        creds_json_str = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')
        creds_info = json.loads(creds_json_str)
        
        credentials = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        gdrive_service = build('drive', 'v3', credentials=credentials)
        log_message("API Google Drive initialisée avec succès.")
        return True
    except json.JSONDecodeError as e:
        log_message(f"ERREUR CRITIQUE: Erreur de décodage JSON des credentials : {e.msg} at pos {e.pos}")
        return False
    except Exception as e:
        log_message(f"ERREUR CRITIQUE: Erreur lors de l'initialisation de l'API Drive : {e}")
        import traceback
        traceback.print_exc()
        return False

def load_yolo_model():
    global yolo_net, yolo_output_layers, yolo_classes
    if not all(os.path.exists(f) for f in [YOLO_WEIGHTS_FILE, YOLO_CFG_FILE, COCO_NAMES_FILE]):
        log_message(f"ERREUR CRITIQUE: Un ou plusieurs fichiers YOLO sont manquants ({YOLO_WEIGHTS_FILE}, {YOLO_CFG_FILE}, {COCO_NAMES_FILE}).")
        return False
    try:
        yolo_net = cv2.dnn.readNet(YOLO_WEIGHTS_FILE, YOLO_CFG_FILE)
        layer_names = yolo_net.getLayerNames()
        
        try:
            unconnected_out_layers = yolo_net.getUnconnectedOutLayers()
            if isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 1:
                yolo_output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
            elif isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 2:
                yolo_output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
            else: 
                yolo_output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        except AttributeError: 
             yolo_output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

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
        return None # None indique une erreur/image invalide
    
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
                return True # Humain détecté
    return False # Aucun humain détecté

def send_email_alert(recipient_email, image_bytes_for_attachment, image_name_for_email):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        log_message("  AVERTISSEMENT: EMAIL_SENDER ou APP_PASSWORD non configurés. Impossible d'envoyer l'email.")
        return

    msg = MIMEMultipart()
    msg['Subject'] = f'🛑 Humain détecté sur l’image: {image_name_for_email}'
    msg['From'] = EMAIL_SENDER
    msg['To'] = recipient_email
    
    body_text = f'Un humain a été détecté sur l’image "{image_name_for_email}" ci-jointe (dossier Google Drive ID: {GDRIVE_FOLDER_ID}).'
    msg.attach(MIMEText(body_text, 'plain'))

    if image_bytes_for_attachment:
        try:
            maintype = 'image'
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

def load_processed_log():
    processed = set()
    if os.path.exists(PROCESSED_LOG_FILE):
        with open(PROCESSED_LOG_FILE, 'r') as f:
            for line in f:
                processed.add(line.strip())
    return processed

def save_to_processed_log(file_id):
    with open(PROCESSED_LOG_FILE, 'a') as f:
        f.write(f"{file_id}\n")

def process_images_from_drive():
    global gdrive_service
    log_message(f"Vérification du dossier Google Drive ID: {GDRIVE_FOLDER_ID}")
    
    processed_file_ids = load_processed_log()
    files_to_process = []

    try:
        page_token = None
        while True:
            results = gdrive_service.files().list(
                q=f"'{GDRIVE_FOLDER_ID}' in parents and (mimeType='image/jpeg' or mimeType='image/png' or mimeType='image/webp') and trashed=false",
                fields="nextPageToken, files(id, name, createdTime)",
                orderBy="createdTime",
                pageSize=100,
                pageToken=page_token
            ).execute()
            
            current_batch = results.get('files', [])
            if not current_batch and page_token is None: # Si premier batch est vide
                log_message("Aucun fichier image trouvé dans le dossier spécifié.")
                break 

            for file_item in current_batch:
                if file_item['id'] not in processed_file_ids:
                    files_to_process.append(file_item)
            
            page_token = results.get('nextPageToken', None)
            if page_token is None:
                break
        
    except Exception as e:
        log_message(f"Erreur lors de la récupération de la liste des fichiers Drive : {e}")
        import traceback
        traceback.print_exc()
        return

    if not files_to_process:
        log_message("Aucune NOUVELLE image à traiter.")
        return

    log_message(f"{len(files_to_process)} nouvelle(s) image(s) trouvée(s) à traiter.")
    
    for file_item in files_to_process:
        file_id = file_item['id']
        file_name = file_item['name']
        log_message(f"  📥 Traitement de {file_name} (ID: {file_id})...")

        image_bytes = None
        img_cv2 = None
        try:
            request = gdrive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk(num_retries=3)
            
            fh.seek(0)
            image_bytes = fh.getvalue()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            detection_result = detect_human(img_cv2) # img_cv2 peut être None si décode échoue

            if detection_result is True:
                log_message(f"  ✅ Humain détecté dans {file_name}. Envoi de l’alerte email.")
                send_email_alert(RECIPIENT_EMAIL, image_bytes, file_name)
            elif detection_result is False:
                log_message(f"  ❌ Aucun humain détecté dans {file_name}.")
            else: # detection_result is None (erreur de décodage ou de détection)
                log_message(f"  ⚠️ Erreur de décodage/détection sur {file_name}. Non traité pour email.")
            
            save_to_processed_log(file_id)

            if DELETE_PROCESSED_FILES:
                try:
                    log_message(f"  🗑️ Tentative de suppression de {file_name} de Google Drive...")
                    gdrive_service.files().delete(fileId=file_id).execute()
                    log_message(f"  Image {file_name} supprimée de Drive avec succès.")
                except Exception as del_e:
                    log_message(f"  ⚠️ Échec de la suppression de {file_name}: {del_e}")

        except Exception as e:
            log_message(f"  Erreur majeure lors du traitement du fichier {file_name} (ID: {file_id}): {e}")
            import traceback
            traceback.print_exc()

# === SCRIPT PRINCIPAL (pour Cron Job) ===
def main():
    log_message("--- Démarrage du script de surveillance Caméra FR (Mode Cron Job) ---")
    
    # Vérification des variables d'environnement avec les noms que vous utilisez
    required_vars = {
        "SENDER_EMAIL": EMAIL_SENDER,
        "APP_PASSWORD": EMAIL_PASSWORD,
        "DEST_EMAIL": RECIPIENT_EMAIL,
        "GOOGLE_CREDENTIALS_BASE64": GOOGLE_CREDENTIALS_BASE64,
        "FOLDER_ID": GDRIVE_FOLDER_ID
    }
    
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        log_message(f"ERREUR CRITIQUE: Variables d'environnement manquantes : {', '.join(missing_vars)}.")
        log_message("Veuillez vérifier leur configuration sur Koyeb.")
        return # Le script s'arrête ici

    if not initialize_drive_api():
        log_message("Échec de l'initialisation de l'API Google Drive. Arrêt.")
        return
    
    if not load_yolo_model():
        log_message("Échec du chargement du modèle YOLO. Arrêt.")
        return

    log_message(f"Dossier Drive surveillé ID: {GDRIVE_FOLDER_ID}")
    log_message(f"Emails envoyés à: {RECIPIENT_EMAIL}")
    log_message(f"Emails envoyés de: {EMAIL_SENDER}")
    log_message(f"Suppression des fichiers après traitement: {DELETE_PROCESSED_FILES}")
    log_message("----------------------------------------------------")
    
    try:
        process_images_from_drive()
    except Exception as e:
        log_message(f"Une erreur majeure est survenue dans main() : {e}")
        import traceback
        traceback.print_exc()
    
    log_message("--- Fin du script de surveillance Caméra FR ---")

if __name__ == "__main__":
    main()
