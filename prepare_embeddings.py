import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re

def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

df = pd.read_csv("zewailcity_faq.csv")
questions = [preprocess_text(q) for q in df['question'].tolist()]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(questions)
np.save("faq_embeddings.npy", embeddings)
print("✅ Embeddings saved.")
