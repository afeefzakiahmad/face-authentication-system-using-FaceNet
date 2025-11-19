from src.preprocess import preprocess_image

import numpy as np


def l2_loss(embedding, database):
    x = np.linalg.norm(embedding - database)
    return x



def getEmbedding(model, image_path):
    face = preprocess_image(image_path)
    emb = model.predict(face, verbose = 0)[0]
    emb = emb/np.linalg.norm(emb)
    return emb




def face_match(model, img1_path, img2_path,):
    embedding1 = getEmbedding(model, img1_path)
    embedding2 = getEmbedding(model, img2_path)
    
    return l2_loss(embedding1, embedding2) < 0.8