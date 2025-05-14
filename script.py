import imaplib
import email
from email.header import decode_header
import os
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2
import datetime

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# Dossier où les images seront enregistrées
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

# Connexion à Gmail
mail = imaplib.IMAP4_SSL("imap.gmail.com")
mail.login(EMAIL, PASSWORD)
mail.select("inbox")

# Cherche les mails non lus contenant les bons mots clés
search_criteria = '(UNSEEN SUBJECT "Alarm event: Motion DetectStart" SUBJECT "Alarm event: Human DetectStart")'
status, messages = mail.search(None, search_criteria)

# Charge le modèle YOLOv8 (vous pouvez utiliser un modèle plus léger si besoin)
model = YOLO("yolov8n.pt")  # Utilise le modèle Nano (rapide). Tu peux essayer yolov8s.pt pour plus de précision.

if status == "OK":
    for num in messages[0].split():
        # Récupère le mail
        status, msg_data = mail.fetch(num, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject = decode_header(msg["Subject"])[0][0]
                if isinstance(subject, bytes):
                    subject = subject.decode()

                print(f"📨 Nouveau mail: {subject}")

                # Parcourt les pièces jointes
                for part in msg.walk():
                    content_dispo = str(part.get("Content-Disposition"))
                    if "attachment" in content_dispo:
                        filename = part.get_filename()
                        if filename and filename.lower().endswith((".jpg", ".jpeg", ".png")):
                            data = part.get_payload(decode=True)
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filepath = os.path.join(SAVE_DIR, f"{timestamp}_{filename}")
                            with open(filepath, "wb") as f:
                                f.write(data)
                            print(f"📸 Image sauvegardée: {filepath}")

                            # YOLOv8 analyse
                            results = model(filepath)
                            annotated_img = results[0].plot()  # Annotated image with boxes

                            # Sauvegarde l'image annotée
                            output_path = filepath.replace(".", "_yolo.")
                            cv2.imwrite(output_path, annotated_img)
                            print(f"✅ Image analysée et enregistrée : {output_path}")
else:
    print("Aucun e-mail avec les mots clés détecté.")

mail.logout()
