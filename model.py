# model.py
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    text = text.lower().strip()
    return re.sub(r'[^\w\s]', '', text)

def get_similarity(user_embedding, all_embeddings):
    sims = cosine_similarity(user_embedding.reshape(1, -1), all_embeddings)
    return sims.flatten()
