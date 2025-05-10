from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from model import preprocess_text, get_similarity
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__, static_folder='static')

faq_data = pd.read_csv("zewailcity_faq.csv")
faq_embeddings = np.load("faq_embeddings.npy")
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    user_embedding = model.encode([preprocess_text(question)])[0]
    sims = get_similarity(user_embedding, faq_embeddings)
    best_idx = np.argmax(sims)
    max_sim = sims[best_idx]

    if max_sim < 0.5:
        return jsonify({"answer": "I'm not sure. Please contact admissions.", "confidence": float(max_sim)})
    
    return jsonify({
        "answer": faq_data['answer'].iloc[best_idx],
        "confidence": float(max_sim)
    })

@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    app.run(debug=True)
