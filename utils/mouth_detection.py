import dlib
import cv2

# Ağız tespiti için dlib modelini kullanacağız
def get_mouth_landmarks(image, face_detector, shape_predictor):
    # Yüz tespiti
    faces = face_detector(image)
    
    for face in faces:
        # Yüz landmarks'larını al
        landmarks = shape_predictor(image, face)
        
        # Ağız landmarks'larını al
        mouth_landmarks = []
        for i in range(48, 68):  # Ağız noktaları dlib'deki numara aralığı
            mouth_landmarks.append((landmarks.part(i).x, landmarks.part(i).y))
        
        return mouth_landmarks
    
    return None

# Konuşma tespiti (ağız hareketi analizi)
def is_speaking(mouth_landmarks):
    if not mouth_landmarks:
        return False
    
    # Ağız açıklığı (örneğin, üst ve alt dudağın arasındaki mesafeyi ölçerek)
    top = mouth_landmarks[2][1]
    bottom = mouth_landmarks[8][1]
    
    mouth_open_ratio = abs(bottom - top)
    
    # Ağız açılma oranına göre konuşma tespiti
    if mouth_open_ratio > 10:
        return True
    return False
