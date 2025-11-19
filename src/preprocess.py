import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN




def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    # img = (img - 127.5) / 128.0
    detector = MTCNN()
    results = detector.detect_faces(img)
    if not results:
        return None
    x,y,w,h = results[0]['box']
    x,y = max(0,x), max(0,y)
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160,160))
    face = face.astype('float32')/255
    face = np.expand_dims(face, axis = 0)
    return face
