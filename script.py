import os
import io
import smtplib
import mimetypes
import cv2
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from email.message import EmailMessage
import time
import base64
import json

# === CONFIGURATION via Variables d'Environnement ===
# Les valeurs par défaut sont fournies pour FOLDER_ID et DEST_EMAIL, 
# mais elles seront écrasées par les variables d'environnement sur Koyeb.
FOLDER_ID = os.environ.get("FOLDER_ID", "1Y-pZkH4S-XvF0UAfl3FmbEGxfGT6_Lxe") 
DEST_EMAIL = os.environ.get("DEST_EMAIL", "jalfatimi@gmail.com")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL") 
APP_PASSWORD = os.environ.get("APP_PASSWORD") 
SCOPES = ['https://www.googleapis.com/auth/drive']

# Gestion de credentials.json via variable d'environnement
GOOGLE_CREDENTIALS_BASE64 = os.environ.get('GOOGLE_CREDENTIALS_BASE64')
if not GOOGLE_CREDENTIALS_BASE64:
    print("ERREUR CRITIQUE: La variable d'environnement GOOGLE_CREDENTIALS_BASE64 n'est pas définie.")
    exit(1) # Quitter si les credentials ne sont pas là

CREDENTIALS_FILE_PATH = '/tmp/credentials.json' 
try:
    creds_json_str = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')
    with open(CREDENTIALS_FILE_PATH, 'w') as f:
        f.write(creds_json_str)
    print("Credentials Google chargés depuis la variable d'environnement.")
except Exception as e:
    print(f"ERREUR CRITIQUE: Erreur lors du décodage ou de l'écriture des credentials : {e}")
    exit(1)

# === INIT DRIVE API ===
try:
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE_PATH, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    print("API Google Drive initialisée avec succès.")
except Exception as e:
    print(f"ERREUR CRITIQUE: Erreur lors de l'initialisation de l'API Drive : {e}")
    exit(1)

# === CHARGER YOLO ===
YOLO_WEIGHTS_PATH = "yolov3-tiny.weights"
YOLO_CFG_PATH = "yolov3-tiny.cfg"

if not os.path.exists(YOLO_WEIGHTS_PATH):
    print(f"ERREUR CRITIQUE: Fichier de poids YOLO non trouvé : {YOLO_WEIGHTS_PATH}")
    exit(1)
if not os.path.exists(YOLO_CFG_PATH):
    print(f"ERREUR CRITIQUE: Fichier de configuration YOLO non trouvé : {YOLO_CFG_PATH}")
    exit(1)

try:
    net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CFG_PATH)
    layer_names = net.getLayerNames()
    # Correction pour les versions d'OpenCV
    try:
        output_layers_indices = net.getUnconnectedOutLayers().flatten()
    except AttributeError: 
        output_layers_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
    print("Modèle YOLO chargé avec succès.")
except Exception as e:
    print(f"ERREUR CRITIQUE: Erreur lors du chargement du modèle YOLO : {e}")
    exit(1)

# === FONCTION : détecter humain ===
def detect_human(image_cv2):
    if image_cv2 is None:
        print("Image non valide reçue pour la détection.")
        return False
    height, width = image_cv2.shape[:2]
    if height == 0 or width == 0:
        print("Image vide reçue pour la détection.")
        return False

    blob = cv2.dnn.blobFromImage(image_cv2, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    try:
        outputs = net.forward(output_layers)
    except Exception as e:
        print(f"Erreur pendant la propagation avant (forward pass) YOLO: {e}")
        return False

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # 0 = person (COCO dataset)
                return True
    return False

# === FONCTION : envoyer mail ===
def send_email_with_image(to_email, image_path, image_name_for_email):
    if not SENDER_EMAIL or not APP_PASSWORD:
        print("AVERTISSEMENT: SENDER_EMAIL ou APP_PASSWORD non configurés. Impossible d'envoyer l'email.")
        return

    msg = EmailMessage()
    msg['Subject'] = f'🛑 Humain détecté sur l’image: {image_name_for_email}'
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg.set_content(f'Un humain a été détecté sur l’image "{image_name_for_email}" ci-jointe (dossier caméra_FR).')

    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        
        ctype, encoding = mimetypes.guess_type(image_path)
        if ctype is None or encoding is not None:
            ctype = 'application/octet-stream' 
        maintype, subtype = ctype.split('/', 1)
        
        msg.add_attachment(img_data, maintype=maintype, subtype=subtype, filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print(f"Email envoyé avec succès à {to_email} pour l'image {image_name_for_email}")
    except FileNotFoundError:
        print(f"Erreur: Fichier image {image_path} non trouvé pour l'envoi par email.")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email : {e}")

# === TRAITER CHAQUE IMAGE ===
def process_images():
    print(f"Vérification du dossier Google Drive ID: {FOLDER_ID}")
    processed_files_count = 0
    try:
        results = service.files().list(
            q=f"'{FOLDER_ID}' in parents and (mimeType='image/jpeg' or mimeType='image/png' or mimeType='image/webp')",
            fields="files(id, name, createdTime)",
            orderBy="createdTime" # Traiter les plus anciens d'abord
        ).execute()
        files = results.get('files', [])
    except Exception as e:
        print(f"Erreur lors de la récupération de la liste des fichiers Drive : {e}")
        return

    if not files:
        print("Aucune nouvelle image trouvée.")
        return

    print(f"{len(files)} image(s) trouvée(s) à traiter.")
    for file_item in files:
        file_id = file_item['id']
        file_name = file_item['name']
        print(f"  📥 Traitement de {file_name} (ID: {file_id})...")

        try:
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            fh.seek(0)
            nparr = np.frombuffer(fh.read(), np.uint8)
            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_cv2 is None:
                print(f"  ❌ Impossible de décoder l'image {file_name}. Suppression de Drive...")
                service.files().delete(fileId=file_id).execute()
                print(f"  Image {file_name} supprimée de Drive.")
                continue

            if detect_human(img_cv2):
                # Créer un nom de fichier temporaire sûr dans /tmp
                # S'assurer que le format est .jpg pour une compatibilité maximale
                base, ext = os.path.splitext(file_name)
                safe_file_name = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in base) # Nettoyer le nom
                temp_image_path = f"/tmp/{safe_file_name}.jpg"
                
                cv2.imwrite(temp_image_path, img_cv2)
                print(f"  ✅ Humain détecté dans {file_name}. Envoi de l’image par mail.")
                send_email_with_image(DEST_EMAIL, temp_image_path, file_name) # Passer le nom original pour le sujet de l'email
                os.remove(temp_image_path) 
            else:
                print(f"  ❌ Aucun humain détecté dans {file_name}.")
            
            print(f"  🗑️ Suppression de {file_name} de Google Drive...")
            service.files().delete(fileId=file_id).execute()
            print(f"  Image {file_name} supprimée de Drive.")
            processed_files_count +=1

        except Exception as e:
            print(f"  Erreur lors du traitement du fichier {file_name} (ID: {file_id}): {e}")
            # Optionnel: supprimer quand même en cas d'erreur pour éviter les boucles
            # try:
            #     service.files().delete(fileId=file_id).execute()
            #     print(f"  Image {file_name} supprimée de Drive après erreur de traitement.")
            # except Exception as del_e:
            #     print(f"  Impossible de supprimer {file_name} après erreur: {del_e}")
    
    if processed_files_count > 0:
        print(f"{processed_files_count} image(s) traitée(s).")


if __name__ == "__main__":
    if not SENDER_EMAIL or not APP_PASSWORD:
        print("AVERTISSEMENT CRITIQUE: SENDER_EMAIL ou APP_PASSWORD ne sont pas définis dans les variables d'environnement. L'envoi d'email ne fonctionnera pas.")
    
    CHECK_INTERVAL_SECONDS = 90 # 1.5 minutes * 60 secondes
    print("--- Démarrage du script de surveillance Caméra FR ---")
    print(f"Vérification toutes les {CHECK_INTERVAL_SECONDS / 60} minutes.")
    print(f"Dossier Drive surveillé ID: {FOLDER_ID}")
    print(f"Emails envoyés à: {DEST_EMAIL}")
    print(f"Emails envoyés de: {SENDER_EMAIL}")
    print("----------------------------------------------------")
    
    while True:
        try:
            process_images()
        except Exception as e:
            print(f"Une erreur majeure est survenue dans la boucle principale : {e}")
            print("Reprise après une courte pause...")
        
        print(f"Prochaine vérification dans {CHECK_INTERVAL_SECONDS / 60:.1f} minutes ({time.strftime('%H:%M:%S', time.localtime(time.time() + CHECK_INTERVAL_SECONDS))}).")
        time.sleep(CHECK_INTERVAL_SECONDS)
