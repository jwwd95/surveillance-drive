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
import json # Importé pour json.loads et json.JSONDecodeError

# === CONFIGURATION via Variables d'Environnement ===
FOLDER_ID = os.environ.get("FOLDER_ID", "1Y-pZkH4S-XvF0UAfl3FmbEGxfGT6_Lxe") 
DEST_EMAIL = os.environ.get("DEST_EMAIL", "jalfatimi@gmail.com")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL") 
APP_PASSWORD = os.environ.get("APP_PASSWORD") 
SCOPES = ['https://www.googleapis.com/auth/drive']

# Gestion de credentials.json via variable d'environnement
GOOGLE_CREDENTIALS_BASE64 = os.environ.get('GOOGLE_CREDENTIALS_BASE64')
if not GOOGLE_CREDENTIALS_BASE64:
    print("ERREUR CRITIQUE: La variable d'environnement GOOGLE_CREDENTIALS_BASE64 n'est pas définie.")
    exit(1)

# === INIT DRIVE API (avec débogage et from_service_account_info) ===
try:
    # 1. Décoder la chaîne Base64
    creds_json_str = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')
    
    # 2. Afficher des parties du JSON décodé pour vérification
    print("--- Contenu JSON décodé (début, max 500 cars) ---")
    print(creds_json_str[:500] + ("..." if len(creds_json_str) > 500 else ""))
    print("--- Contenu JSON décodé (fin, max 500 cars) ---")
    if len(creds_json_str) > 500:
        print("..." + creds_json_str[-500:])
    else:
        print(creds_json_str) # Si plus court, afficher tout
    
    # 3. Parser le JSON en un dictionnaire Python
    creds_info = json.loads(creds_json_str) 
    
    # 4. Vérifier et afficher le début de la clé privée
    if 'private_key' in creds_info:
        print("--- Clé privée extraite (début, max 100 cars) ---")
        pk_str = creds_info['private_key']
        print(pk_str[:100] + ("..." if len(pk_str) > 100 else ""))
        print("--- Clé privée extraite (fin, max 100 cars) ---")
        if len(pk_str) > 100:
            print("..." + pk_str[-100:])
        else:
            print(pk_str)
    else:
        print("AVERTISSEMENT CRITIQUE: 'private_key' non trouvée dans les credentials JSON décodés!")
        exit(1) # Quitter si la clé privée est manquante

    # 5. Charger les credentials directement depuis le dictionnaire
    credentials = service_account.Credentials.from_service_account_info(
        creds_info, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    print("API Google Drive initialisée avec succès (via from_service_account_info).")

except json.JSONDecodeError as e:
    print(f"ERREUR CRITIQUE: Erreur de décodage JSON des credentials : {e}")
    print(f"  Message: {e.msg}")
    print(f"  Document (extrait autour de l'erreur): '{e.doc[max(0,e.pos-30):e.pos+30]}'")
    print(f"  Position: ligne {e.lineno}, colonne {e.colno} (index {e.pos})")
    exit(1)
except Exception as e:
    print(f"ERREUR CRITIQUE: Erreur lors du chargement/initialisation des credentials ou de l'API Drive : {e}")
    # Afficher la trace complète pour plus de détails en cas d'erreur inattendue
    import traceback
    traceback.print_exc()
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
            if class_id == 0 and confidence > 0.5:
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
            orderBy="createdTime"
        ).execute()
        files = results.get('files', [])
    except Exception as e:
        print(f"Erreur lors de la récupération de la liste des fichiers Drive : {e}")
        # Afficher la trace complète pour plus de détails en cas d'erreur
        import traceback
        traceback.print_exc()
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
                base, ext = os.path.splitext(file_name)
                safe_file_name = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in base)
                temp_image_path = f"/tmp/{safe_file_name}.jpg" # Utiliser /tmp, inscriptible dans Docker
                
                cv2.imwrite(temp_image_path, img_cv2)
                print(f"  ✅ Humain détecté dans {file_name}. Envoi de l’image par mail.")
                send_email_with_image(DEST_EMAIL, temp_image_path, file_name)
                os.remove(temp_image_path) 
            else:
                print(f"  ❌ Aucun humain détecté dans {file_name}.")
            
            print(f"  🗑️ Suppression de {file_name} de Google Drive...")
            service.files().delete(fileId=file_id).execute()
            print(f"  Image {file_name} supprimée de Drive.")
            processed_files_count +=1

        except Exception as e:
            print(f"  Erreur lors du traitement du fichier {file_name} (ID: {file_id}): {e}")
            import traceback
            traceback.print_exc()
    
    if processed_files_count > 0:
        print(f"{processed_files_count} image(s) traitée(s).")

if __name__ == "__main__":
    if not SENDER_EMAIL or not APP_PASSWORD:
        print("AVERTISSEMENT CRITIQUE: SENDER_EMAIL ou APP_PASSWORD ne sont pas définis. L'envoi d'email ne fonctionnera pas.")
    
    CHECK_INTERVAL_SECONDS = 90
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
            import traceback
            traceback.print_exc()
            print("Reprise après une courte pause...")
        
        current_time_plus_interval = time.time() + CHECK_INTERVAL_SECONDS
        print(f"Prochaine vérification dans {CHECK_INTERVAL_SECONDS / 60:.1f} minutes (vers {time.strftime('%H:%M:%S', time.localtime(current_time_plus_interval))}).")
        time.sleep(CHECK_INTERVAL_SECONDS)
