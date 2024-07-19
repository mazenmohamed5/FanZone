import cv2
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import face_recognition
import urllib.request
import time
from Crypto.Cipher import AES
import base64
from mock_gpio import MockGPIO as GPIO

# Initialize Firebase
cred = credentials.Certificate("")
firebase_admin.initialize_app(cred, {''})
db = firestore.client()

# Encryption key 
encryption_key = b'Sixteen byte key'

def encrypt_data(data, key):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return base64.b64encode(nonce + ciphertext).decode('utf-8')

def decrypt_data(encrypted_data, key):
    decoded_data = base64.b64decode(encrypted_data)
    nonce = decoded_data[:16]
    ciphertext = decoded_data[16:]
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    return plaintext.decode('utf-8')



# GPIO setup
RECOGNIZED_PIN = 17  # GPIO pin to set high if face is recognized
NOT_RECOGNIZED_PIN = 27  # GPIO pin to set high if face is not recognized

GPIO.setmode(GPIO.BCM)  #
GPIO.setup(RECOGNIZED_PIN, GPIO.OUT)
GPIO.setup(NOT_RECOGNIZED_PIN, GPIO.OUT)

def send_qr_data_to_laptop(qr_data):
    server_url = "http://<Laptop_IP_Address>:5000/scan_qr"  # laptop's IP address
    encrypted_qr_data = encrypt_data(qr_data, encryption_key)
    response = requests.post(server_url, json={'match_doc_id': encrypted_qr_data})
    return response.json()

def receive_recognition_result():
    server_url = "http://<Laptop_IP_Address>:5000/get_recognition_result"  # laptop's IP address
    response = requests.get(server_url)
    return response.json()

class FirebaseImageLoader:
    def __init__(self):
        self.db = firestore.client()

    def get_image_by_id(self, doc_id, collection, is_dependent=False):
        images_ref = self.db.collection(collection)
        user_doc = images_ref.document(doc_id).get()

        if user_doc.exists:
            image_url_field = None
            if collection == 'Fan' and not is_dependent:
                image_url_field = 'fanImageURL'
            elif collection == 'Family_Members' and is_dependent:
                image_url_field = 'depImageURL'
            else:
                print(f"Invalid collection or ID type: {collection}, is_dependent: {is_dependent}")
                return None
            
            fan_image_url = user_doc.to_dict().get(image_url_field)
            if fan_image_url:
                try:
                    with urllib.request.urlopen(fan_image_url) as response:
                        image_data = response.read()
                    return image_data
                except Exception as e:
                    print(f"Error fetching image from URL: {e}")
                    return None
            else:
                print(f"No {image_url_field} found for document ID: {doc_id}")
                return None
        else:
            print(f"No document found for ID: {doc_id} in collection: {collection}")
            return None

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.tolerance = 0.4
        self.firebase_loader = FirebaseImageLoader()

    def load_user_encoding_images(self, user_id):
        if user_id is not None:
            image_data = self.firebase_loader.get_image_by_id(user_id, collection='Fan')
            if image_data is not None:
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                self.known_face_encodings.append(img_encoding)
                user_doc = db.collection('Fan').document(user_id).get()
                if user_doc.exists:
                    self.known_face_names.append(user_doc.get('fullname'))
                else:
                    print(f"No document found for user ID {user_id} in the Fan collection.")
                    self.known_face_names.append("Unknown")
            print('User encoding images loaded')
        else:
            print("No user ID provided.")

    def load_dependent_encoding_images(self, dep_ids):
        if dep_ids:
            for dep_id in dep_ids:
                image_data = self.firebase_loader.get_image_by_id(dep_id, collection='Family_Members', is_dependent=True)
                if image_data is not None:
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_encoding = face_recognition.face_encodings(rgb_img)[0]
                    self.known_face_encodings.append(img_encoding)
                    dep_doc = db.collection('Family_Members').document(dep_id).get()
                    if dep_doc.exists:
                        self.known_face_names.append(dep_doc.get('depName'))
                    else:
                        print(f"No document found for dependent ID {dep_id} in the Family_Members collection.")
                        self.known_face_names.append("Unknown")
            print('Dependent encoding images loaded')
        else:
            print("No dependent IDs provided.")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_loc_names = []
        for face_encoding, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
            name = 'Unknown'

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            top, right, bottom, left = face_loc
            top *= int(1/self.frame_resizing)
            right *= int(1/self.frame_resizing)
            bottom *= int(1/self.frame_resizing)
            left *= int(1/self.frame_resizing)

            face_names.append(name)
            face_loc_names.append(((top, right, bottom, left), name))

        return face_locations, face_names, face_loc_names

if __name__ == "__main__":
    try:
        while True:
            qr_data = input("Scan QR code (type 'exit' to quit): ")
            if qr_data.lower() == 'exit':
                print("Exiting QR code scanning...")
                break

            response_data = send_qr_data_to_laptop(qr_data)
            if 'error' not in response_data:
                print("User ID and Dependent IDs received from the laptop for face recognition.")
                user_id = response_data['user_id']
                dep_ids = response_data['dep_ids']

                face_recognizer = SimpleFacerec()
                face_recognizer.load_user_encoding_images(user_id)
                face_recognizer.load_dependent_encoding_images(dep_ids)

                cap = cv2.VideoCapture(0)
                user_recognized = False
                dependents_recognized = {face_recognizer.known_face_names[i]: False for i in range(1, len(face_recognizer.known_face_names))}
                start_time = None

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Unable to capture frame from the camera")
                        break

                    face_locations, face_names, face_loc_names = face_recognizer.detect_known_faces(frame)
                    for face_loc, name in face_loc_names:
                        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                    if user_id and not user_recognized and face_recognizer.known_face_names[0] in face_names:
                        user_recognized = True
                        print("User recognized")

                    for name in face_names:
                        if name in dependents_recognized and not dependents_recognized[name]:
                            dependents_recognized[name] = True
                            print(f"Dependent {name} recognized")

                    if user_recognized and all(dependents_recognized.values()):
                        message = "Welcome to the stadium, " + face_recognizer.known_face_names[0] + " and dependents!"
                        if not start_time:
                            start_time = time.time()

                    cv2.imshow('Face Recognition', frame)

                    key = cv2.waitKey(1)
                    if key == 27:
                        break

                    if start_time and (time.time() - start_time >= 5):
                        cv2.destroyAllWindows()
                        print("Recognition result: ", message)
                        GPIO.output(RECOGNIZED_PIN, GPIO.HIGH)
                        GPIO.output(NOT_RECOGNIZED_PIN, GPIO.LOW)
                        cap.release()
                        break

                cap.release()
                cv2.destroyAllWindows()
            else:
                print("Error: ", response_data['error'])
                GPIO.output(RECOGNIZED_PIN, GPIO.LOW)
                GPIO.output(NOT_RECOGNIZED_PIN, GPIO.HIGH)
    finally:
        GPIO.cleanup()  # Ensure GPIO pins are reset on exit
