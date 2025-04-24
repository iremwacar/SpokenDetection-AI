import cv2

# Yüz tanıma için önceden eğitilmiş sınıflandırıcıyı yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamera aç
cap = cv2.VideoCapture(0)

# Sonsuz döngü ile her bir kareyi al
while True:
    ret, frame = cap.read()  # Kamera görüntüsünü oku
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Görüntüyü gri tonlamaya çevir

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Yüzleri dikdörtgen ile çiz
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Yüzü çiz

    # Görüntüyü ekranda göster
    cv2.imshow('Video', frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kapama
cap.release()
cv2.destroyAllWindows()
