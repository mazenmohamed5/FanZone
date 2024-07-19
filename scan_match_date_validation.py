import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import face_recognition
import urllib.request
import time
from datetime import datetime, timedelta

# Initialize Firebase
cred = credentials.Certificate("")
firebase_admin.initialize_app(cred, {''})
db = firestore.client()

# Class for loading images from Firebase Firestore
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

# Class for face recognition
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

def display_face_recognition_window(frame):
    cv2.imshow('Face Recognition', frame)

def get_user_and_dependent_id_from_ticketsto(match_doc_id):
    match_ref = db.collection('Match_Tickets').document(match_doc_id)
    match_doc = match_ref.get()
    if match_doc.exists:
        ticket_status = match_doc.get('ticketStatus')
        if ticket_status != 'Activated':
            print("Ticket is not activated.")
            return None, [], None, None, None

        match_date = match_doc.get('matchDate')
        match_time = match_doc.get('matchTime')

        if not match_date or not match_time:
            print("Match date or time not specified.")
            return None, [], None, None, None

        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().time()

        if current_date != match_date:
            print("Today is not the match date.")
            return None, [], None, None, None

        match_datetime = datetime.strptime(f"{match_date} {match_time}", '%Y-%m-%d %H:%M')
        end_time = match_datetime + timedelta(hours=2)

        if current_time < match_datetime.time() or current_time > end_time.time():
            print("Current time is outside the match window.")
            return None, [], None, None, None

        ticketsto = match_doc.get('TicketsTo')
        if ticketsto:
            user_id = ticketsto[0] if len(ticketsto) >= 1 else None
            dep_ids = ticketsto[1:] if len(ticketsto) > 1 else []
            return user_id, dep_ids, match_ref, match_date, match_time
        else:
            return None, [], None, None, None
    else:
        return None, [], None, None, None

def update_ticket_status(match_ref):
    try:
        match_ref.update({"ticketStatus": "Boarded"})
        print("Ticket status updated to 'Boarded'.")
    except Exception as e:
        print(f"Error updating ticket status: {e}")

def display_welcome_message(message):
    welcome_window = np.zeros((200, 800, 3), dtype=np.uint8)
    font_scale = 0.5
    font_thickness = 1
    font_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, font_scale, font_thickness)[0]
    text_x = int((welcome_window.shape[1] - text_size[0]) / 2)
    text_y = int((welcome_window.shape[0] + text_size[1]) / 2)
    cv2.putText(welcome_window, message, (text_x, text_y), font, font_scale, font_color, font_thickness)
    cv2.imshow("Welcome", welcome_window)

cap_main = cv2.VideoCapture(0)
main_frame_width = 800
main_frame_height = 600
cap_main.set(cv2.CAP_PROP_FRAME_WIDTH, main_frame_width)
cap_main.set(cv2.CAP_PROP_FRAME_HEIGHT, main_frame_height)

while True:
    match_doc_id = input("Scan QR code (type 'exit' to quit): ")
    if match_doc_id.lower() == 'exit':
        print("Exiting QR code scanning...")
        break

    user_id, dep_ids, match_ref, match_date, match_time = get_user_and_dependent_id_from_ticketsto(match_doc_id)
    if user_id or dep_ids:
        face_recognizer = SimpleFacerec()
        if user_id:
            face_recognizer.load_user_encoding_images(user_id)
        if dep_ids:
            face_recognizer.load_dependent_encoding_images(dep_ids)
        
        user_recognized = False
        dependents_recognized = {dep_id: False for dep_id in dep_ids}
        start_time = None

        while True:
            ret, frame = cap_main.read()
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
                if name in face_recognizer.known_face_names[1:] and not dependents_recognized[name]:
                    dependents_recognized[name] = True
                    print(f"Dependent {name} recognized")

            if user_recognized and all(dependents_recognized.values()):
                message = "Welcome to the stadium, " + face_recognizer.known_face_names[0] + " and dependents!"
                display_welcome_message(message)
                if not start_time:
                    start_time = time.time()

            display_face_recognition_window(frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

            if start_time and (time.time() - start_time >= 5):
                cv2.destroyAllWindows()
                update_ticket_status(match_ref)
                break

cap_main.release()
cv2.destroyAllWindows()
