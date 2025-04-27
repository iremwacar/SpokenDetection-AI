import cv2
import dlib
import numpy as np
import face_recognition
from utils.mouth_detection import get_mouth_landmarks, is_speaking

# Yüz algılayıcı ve landmark modeli
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Video yakalama
cap = cv2.VideoCapture(0)

known_face_encodings = []
known_face_ids = []
face_timers = {}
next_face_id = 0

frame_count = 0
current_faces = []
current_speaker_id = None

cv2.namedWindow('Spoken Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü büyütelim ki küçük yüzleri kaybetmesin
    big_frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    gray_big = cv2.cvtColor(big_frame, cv2.COLOR_BGR2GRAY)

    if frame_count % 3 == 0:
        faces = face_detector(gray_big, 1)  # upsample_num_times = 1
        current_faces = []

        for face in faces:
            (x, y, x2, y2) = (face.left(), face.top(), face.right(), face.bottom())
            x, y, x2, y2 = int(x / 1.5), int(y / 1.5), int(x2 / 1.5), int(y2 / 1.5)  # koordinatları küçült
            face_img = frame[y:y2, x:x2]
            if face_img.size == 0:
                continue

            face_encoding = face_recognition.face_encodings(frame, [(y, x2, y2, x)])
            if len(face_encoding) == 0:
                continue
            face_encoding = face_encoding[0]

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
            else:
                best_match_index = None

            if best_match_index is not None and matches[best_match_index]:
                face_id = known_face_ids[best_match_index]
            else:
                face_id = next_face_id
                next_face_id += 1
                known_face_encodings.append(face_encoding)
                known_face_ids.append(face_id)
                face_timers[face_id] = 0

            current_faces.append((face_id, (x, y, x2, y2)))

    frame_count += 1

    speaking_now = None

    for face_id, (left, top, right, bottom) in current_faces:
        rect = dlib.rectangle(left, top, right, bottom)
        mouth_points = get_mouth_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), rect, shape_predictor)

        if is_speaking(mouth_points):
            speaking_now = face_id
            break

    for face_id, (left, top, right, bottom) in current_faces:
        if face_id == speaking_now:
            color = (0, 255, 0)  # Yeşil: konuşuyor
            face_timers[face_id] += 1
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"ID: {face_id} - {face_timers[face_id]}s", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Spoken Detection', frame)

    key = cv2.waitKey(10) & 0xFF

    # 🚨 X tuşu veya pencere kapatma kontrolü
    if key == ord('x') or key == ord('X'):
        print("X'e basıldı, çıkılıyor...")
        break

    if cv2.getWindowProperty('Spoken Detection', cv2.WND_PROP_VISIBLE) < 1:
        print("Pencere kapandı, çıkılıyor...")
        break

# 🛑 Pencere kapanınca terminale yazdır
print("\nKonuşma Süreleri:")
for face_id, seconds in face_timers.items():
    print(f"ID {face_id}: {seconds} saniye konuştu.")

cap.release()
cv2.destroyAllWindows()
