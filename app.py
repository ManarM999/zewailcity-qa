# app.py
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from model import preprocess_text, get_similarity
import os
from sentence_transformers import SentenceTransformer

app = Flask(__name__, static_folder='static')

# Load data
faq_data = pd.read_csv("zewailcity_faq.csv")
faq_embeddings = np.load("faq_embeddings.npy")

# Load model once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_question = data.get("question", "").strip()
        if not user_question:
            return jsonify({"answer": "Please enter a valid question."}), 400

        processed = preprocess_text(user_question)
        user_embedding = model.encode([processed])[0]  # Shape (384,)

        similarities = get_similarity(user_embedding, faq_embeddings)
        best_idx = np.argmax(similarities)
        max_sim = similarities[best_idx]

        if max_sim < 0.5:
            return jsonify({
                "answer": "I'm not sure about that. Please contact admissions.",
                "confidence": float(max_sim)
            })

        return jsonify({
            "answer": faq_data['answer'].iloc[best_idx],
            "confidence": float(max_sim)
        })

    except Exception as e:
        return jsonify({"answer": "An error occurred: " + str(e)}), 500

@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    app.run(debug=True)
