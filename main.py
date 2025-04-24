import cv2
from utils.face_detection import detect_face
from utils.mouth_detection import get_mouth_landmarks, is_speaking

# Kamera aç
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti
    faces = detect_face(gray)

    for face in faces:
        # Yüz bölgesini çiz
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Ağız noktalarını al
        mouth_points = get_mouth_landmarks(frame, face)

        # Ağız açılma oranını hesapla ve konuşma durumunu kontrol et
        if is_speaking(mouth_points):
            cv2.putText(frame, "Konusuyor...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Face and Mouth Detection", frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
