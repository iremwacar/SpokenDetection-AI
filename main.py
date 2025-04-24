import dlib
import cv2
from utils.face_detection import detect_face
from utils.mouth_detection import get_mouth_landmarks, is_speaking

# Dlib yüz tespiti için face_detector ve shape_predictor kullan
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Video veya webcam kullan
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Yüz tespiti yap (face_detector'ı burada kullanmaya gerek yok, detect_face fonksiyonu zaten bunu yapmalı)
    faces = detect_face(frame)

    for face in faces:
        # Ağız noktalarını al, shape_predictor'ı da geçiyoruz
        mouth_points = get_mouth_landmarks(frame, face, shape_predictor)

        # Konuşma tespiti
        if is_speaking(mouth_points):
            print("Konuşma Tespit Edildi")
        else:
            print("Konuşma Tespit Edilmedi")

    # Çerçeveyi göster
    cv2.imshow("Frame", frame)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
