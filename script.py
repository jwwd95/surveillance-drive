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

# === CONFIGURATION ===
FOLDER_ID = "1Y-pZkH4S-XvF0UAfl3FmbEGxfGT6_Lxe"
DEST_EMAIL = "jalfatimi@gmail.com"
SENDER_EMAIL = "saidben9560@gmail.com"
APP_PASSWORD = "ajut jinq dwkp pywj"
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'surveillancedrive-93dee7913b77.json'

# === INIT DRIVE API ===
credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

# === CHARGER YOLO ===
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# === FONCTION : détecter humain ===
def detect_human(image):
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # 0 = person
                return True
    return False

# === FONCTION : envoyer mail ===
def send_email_with_image(to_email, image_path):
    msg = EmailMessage()
    msg['Subject'] = '🛑 Humain détecté sur l’image'
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg.set_content('Un humain a été détecté sur l’image ci-jointe.')

    with open(image_path, 'rb') as f:
        img_data = f.read()
        mime_type, _ = mimetypes.guess_type(image_path)
        maintype, subtype = mime_type.split('/')
        msg.add_attachment(img_data, maintype=maintype, subtype=subtype, filename=os.path.basename(image_path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)

# === TRAITER CHAQUE IMAGE ===
def process_images():
    results = service.files().list(q=f"'{FOLDER_ID}' in parents and mimeType contains 'image/'",
                                   fields="files(id, name)").execute()
    files = results.get('files', [])

    for file in files:
        file_id = file['id']
        file_name = file['name']
        print(f"📥 Téléchargement de {file_name}...")

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        fh.seek(0)
        nparr = np.frombuffer(fh.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if detect_human(img):
            temp_path = f"/tmp/{file_name}"
            cv2.imwrite(temp_path, img)
            print("✅ Humain détecté, envoi de l’image par mail.")
            send_email_with_image(DEST_EMAIL, temp_path)
            os.remove(temp_path)
        else:
            print("❌ Aucun humain détecté, suppression de l’image.")
        
        service.files().delete(fileId=file_id).execute()

if __name__ == "__main__":
    process_images()
