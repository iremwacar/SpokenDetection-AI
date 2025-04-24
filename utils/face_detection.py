import dlib

# Yüz tanıma için Dlib'in yüz tespitini yükle
detector = dlib.get_frontal_face_detector()

def detect_face(gray_image):
    faces = detector(gray_image)
    return faces
