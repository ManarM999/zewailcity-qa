from sentence_transformers import SentenceTransformer
import re

_model = None

def preprocess_text(text):
    text = text.lower().strip()
    return re.sub(r'[^\w\s]', '', text)  # Remove punctuation

def get_embeddings(texts):
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient & accurate
    
    processed_texts = [preprocess_text(t) for t in texts]
    embeddings = _model.encode(processed_texts, convert_to_tensor=True)
    return embeddings