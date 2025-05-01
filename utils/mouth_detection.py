import dlib
import numpy as np

# shape_predictor modelini yükle
def get_mouth_landmarks(image, face, shape_predictor):
    # Dlib şekil tespitinden yüz noktalarını al
    landmarks = shape_predictor(image, face)
    
    # Ağız noktalarını döndür
    mouth_points = []
    for i in range(48, 68):  # Ağız noktaları 48-67 arasında
        mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))
    return mouth_points

def is_speaking(mouth_points, mode="near"):
    # Ağız genişliği (ağız noktalarındaki x mesafesi)
    mouth_width = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))
    
    # Ağız yüksekliği (ağız noktalarındaki y mesafesi)
    mouth_height = np.linalg.norm(np.array(mouth_points[3]) - np.array(mouth_points[9]))

    # Eşik değerlerini belirleyelim (yakın ve uzak için farklı)
    if mode == "near":
        # Yakın mesafede daha küçük ağız hareketlerine duyarlılık
        width_threshold = 70  # Ağız genişliği için daha büyük eşik
        height_threshold = 50  # Ağız yüksekliği için daha büyük eşik
    elif mode == "far":
        width_threshold = 45  # Uzak mesafede daha büyük ağız hareketlerine duyarlılık
        height_threshold = 35
    else:
        width_threshold = 60  # Varsayılan eşik değerleri
        height_threshold = 45

    # Konuşma olup olmadığını belirlemek için ağız genişliğini ve yüksekliğini kontrol et
    if mouth_width > width_threshold and mouth_height > height_threshold:
        return True
    return False

