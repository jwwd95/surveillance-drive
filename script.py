import os
import io
import smtplib
import mimetypes
import cv2
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from email.message import EmailMessage # Vous utilisiez ceci, c'est bien
from email.mime.image import MIMEImage # Alternative pour attachement direct
from email.mime.multipart import MIMEMultipart # Nécessaire si on combine texte et image
from email.mime.text import MIMEText      # Nécessaire pour le corps du texte avec MIMEMultipart
import time
import base64
import json
import datetime # Pour les logs et le processed_log

# === CONFIGURATION via Variables d'Environnement ===
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "1Y-pZkH4S-XvF0UAfl3FmbEGxfGT6_Lxe") 
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "JalFatimi@gmail.com") # Renommé pour la cohérence
EMAIL_SENDER = os.environ.get("EMAIL_SENDER") 
EMAIL_PASSWORD = os.environ.get("APP_PASSWORD") # APP_PASSWORD est le bon nom que vous utilisiez
SCOPES = ['https://www.googleapis.com/auth/drive'] # Accès complet nécessaire pour la suppression

GOOGLE_CREDENTIALS_BASE64 = os.environ.get('GOOGLE_CREDENTIALS_BASE64')

# Comportement de suppression
DELETE_PROCESSED_FILES = os.environ.get('DELETE_PROCESSED_FILES', 'true').lower() == 'true'

# Fichiers YOLO
YOLO_WEIGHTS_FILE = "yolov3-tiny.weights"
YOLO_CFG_FILE = "yolov3-tiny.cfg"
COCO_NAMES_FILE = "coco.names" # Assurez-vous que ce fichier existe et contient les classes, "person" doit être dedans.

PROCESSED_LOG_FILE = "processed_log.txt" # Pour se souvenir des fichiers traités

# Variables globales pour le service Drive et le modèle YOLO
gdrive_service = None
yolo_net = None
yolo_output_layers = None
yolo_classes = None # Les noms des classes YOLO

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
        # log_message("--- Contenu JSON décodé (extrait) ---")
        # log_message(creds_json_str[:200] + "..." + creds_json_str[-200:]) # Log plus concis
        creds_info = json.loads(creds_json_str)
        
        # if 'private_key' in creds_info:
        #     log_message("--- Clé privée extraite (info de présence uniquement) ---")
        # else:
        #     log_message("AVERTISSEMENT: 'private_key' non trouvée dans les credentials JSON décodés!")
        #     return False

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
        
        # Gestion de getUnconnectedOutLayers pour différentes versions d'OpenCV
        try:
            unconnected_out_layers = yolo_net.getUnconnectedOutLayers()
            if isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 1: # Ex: array([200, 227, 254])
                yolo_output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
            elif isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 2: # Ex: array([[200], [227], [254]])
                yolo_output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
            else: # Fallback si c'est une liste d'indices ou autre format
                yolo_output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        except AttributeError: # Pour les très anciennes versions d'OpenCV
             yolo_output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]


        with open(COCO_NAMES_FILE, 'r') as f:
            yolo_classes = [line.strip() for line in f.readlines()]
        if 'person' not in yolo_classes:
            log_message("ERREUR CRITIQUE: La classe 'person' n'est pas trouvée dans coco.names. La détection d'humain échouera.")
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
        return False # False pour "pas d'humain", None pour "erreur de détection"
    
    height, width = image_cv2.shape[:2]
    if height == 0 or width == 0:
        log_message("  Image vide reçue pour la détection.")
        return False

    blob = cv2.dnn.blobFromImage(image_cv2, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    try:
        outputs = yolo_net.forward(yolo_output_layers)
    except Exception as e:
        log_message(f"  Erreur pendant la propagation avant (forward pass) YOLO: {e}")
        return None # Indique une erreur pendant la détection

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id_index = np.argmax(scores)
            confidence = scores[class_id_index]
            # Assurez-vous que class_id_index est dans les limites de yolo_classes
            if class_id_index < len(yolo_classes) and yolo_classes[class_id_index] == 'person' and confidence > 0.5:
                return True # Humain détecté
    return False # Aucun humain détecté

def send_email_alert(recipient_email, image_bytes_for_attachment, image_name_for_email):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        log_message("  AVERTISSEMENT: EMAIL_SENDER ou EMAIL_PASSWORD non configurés. Impossible d'envoyer l'email.")
        return

    # Utiliser MIMEMultipart pour une meilleure compatibilité et structure
    msg = MIMEMultipart()
    msg['Subject'] = f'🛑 Humain détecté sur l’image: {image_name_for_email}'
    msg['From'] = EMAIL_SENDER
    msg['To'] = recipient_email
    
    body_text = f'Un humain a été détecté sur l’image "{image_name_for_email}" ci-jointe (dossier Google Drive ID: {GDRIVE_FOLDER_ID}).'
    msg.attach(MIMEText(body_text, 'plain'))

    if image_bytes_for_attachment:
        try:
            # Déterminer le type MIME principal et secondaire
            # Pour les images JPEG ou PNG, c'est simple
            maintype = 'image'
            subtype = 'jpeg' if image_name_for_email.lower().endswith(('.jpg', '.jpeg')) else 'png'
            # Si vous voulez être plus générique (moins sûr):
            # ctype, _ = mimetypes.guess_type(image_name_for_email)
            # if ctype is None or ctype.split('/')[0] != 'image':
            #     maintype, subtype = 'application', 'octet-stream' # Fallback
            # else:
            #     maintype, subtype = ctype.split('/', 1)
            
            img_mime = MIMEImage(image_bytes_for_attachment, subtype=subtype, name=os.path.basename(image_name_for_email))
            img_mime.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_name_for_email))
            msg.attach(img_mime)
        except Exception as e:
            log_message(f"  Erreur lors de l'attachement de l'image {image_name_for_email}: {e}")
            # Continuer sans l'image si l'attachement échoue
    
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
                orderBy="createdTime", # Traiter les plus anciens en premier
                pageSize=100, # Récupérer par lots
                pageToken=page_token
            ).execute()
            
            current_batch = results.get('files', [])
            if not current_batch:
                log_message("Aucun fichier image trouvé dans le dossier spécifié.")
                break # Sortir si le dossier est vide dès le départ

            for file_item in current_batch:
                if file_item['id'] not in processed_file_ids:
                    files_to_process.append(file_item)
            
            page_token = results.get('nextPageToken', None)
            if page_token is None:
                break # Fin de la liste des fichiers
        
    except Exception as e:
        log_message(f"Erreur lors de la récupération de la liste des fichiers Drive : {e}")
        import traceback
        traceback.print_exc()
        return

    if not files_to_process:
        log_message("Aucune NOUVELLE image à traiter (toutes déjà dans processed_log ou dossier vide).")
        return

    log_message(f"{len(files_to_process)} nouvelle(s) image(s) trouvée(s) à traiter.")
    
    for file_item in files_to_process:
        file_id = file_item['id']
        file_name = file_item['name']
        log_message(f"  📥 Traitement de {file_name} (ID: {file_id})...")

        image_bytes = None
        try:
            request = gdrive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk(num_retries=3) # Ajout de reintentions
            
            fh.seek(0)
            image_bytes = fh.getvalue()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_cv2 is None:
                log_message(f"  ❌ Impossible de décoder l'image {file_name}. Marquage comme traité.")
                save_to_processed_log(file_id) # Marquer pour ne pas réessayer
                if DELETE_PROCESSED_FILES:
                    try:
                        gdrive_service.files().delete(fileId=file_id).execute()
                        log_message(f"  🗑️ Image corrompue/indécodable {file_name} supprimée de Drive.")
                    except Exception as del_e:
                        log_message(f"  ⚠️ Échec de la suppression de l'image corrompue {file_name}: {del_e}")
                continue

            detection_result = detect_human(img_cv2)

            if detection_result is True: # Humain détecté
                log_message(f"  ✅ Humain détecté dans {file_name}. Envoi de l’alerte email.")
                send_email_alert(RECIPIENT_EMAIL, image_bytes, file_name)
            elif detection_result is False: # Aucun humain détecté
                log_message(f"  ❌ Aucun humain détecté dans {file_name}.")
            else: # Erreur pendant la détection (detection_result is None)
                log_message(f"  ⚠️ Erreur lors de la détection sur {file_name}. Non traité pour email.")
            
            save_to_processed_log(file_id) # Marquer comme traité (que ce soit avec ou sans humain, ou erreur de détection)

            if DELETE_PROCESSED_FILES:
                try:
                    log_message(f"  🗑️ Tentative de suppression de {file_name} de Google Drive...")
                    gdrive_service.files().delete(fileId=file_id).execute()
                    log_message(f"  Image {file_name} supprimée de Drive avec succès.")
                except Exception as del_e:
                    log_message(f"  ⚠️ Échec de la suppression de {file_name} après traitement: {del_e}")
                    import traceback
                    traceback.print_exc() # Loguer l'erreur complète de suppression

        except Exception as e:
            log_message(f"  Erreur majeure lors du traitement du fichier {file_name} (ID: {file_id}): {e}")
            import traceback
            traceback.print_exc()
            # Ne pas marquer comme traité pour qu'il soit réessayé, sauf si c'est une erreur de téléchargement persistante.
            # Ou alors, avoir un compteur de tentatives dans processed_log. Pour l'instant, on le laisse pour un prochain essai.

# === SCRIPT PRINCIPAL (pour Cron Job) ===
def main():
    log_message("--- Démarrage du script de surveillance Caméra FR (Mode Cron Job) ---")
    
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, RECIPIENT_EMAIL, GOOGLE_CREDENTIALS_BASE64, GDRIVE_FOLDER_ID]):
        log_message("ERREUR CRITIQUE: Variables d'environnement manquantes. Vérifiez EMAIL_SENDER, APP_PASSWORD, RECIPIENT_EMAIL, GOOGLE_CREDENTIALS_BASE64, GDRIVE_FOLDER_ID.")
        return

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
