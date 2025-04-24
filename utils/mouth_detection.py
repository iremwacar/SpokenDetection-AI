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

def is_speaking(mouth_points):
    # Ağız genişliği (ağız noktalarındaki x mesafesi)
    mouth_width = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))
    
    # Ağız yüksekliği (ağız noktalarındaki y mesafesi)
    mouth_height = np.linalg.norm(np.array(mouth_points[3]) - np.array(mouth_points[9]))

    # Konuşma olup olmadığını belirlemek için ağız genişliğini ve yüksekliğini kontrol et
    if mouth_width > 55 and mouth_height > 40:  # Bu değerleri ihtiyaca göre ayarlayın
        return True
    return False
