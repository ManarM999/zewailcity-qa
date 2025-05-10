from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import torch
import os
from model import get_embeddings

app = Flask(__name__, static_folder='static')

# Configuration
MIN_SIMILARITY = 0.5  

# Load FAQ data
try:
    faq_data = pd.read_csv('zewailcity_faq.csv')
    faq_embeddings = get_embeddings(faq_data['question'].tolist())
except Exception as e:
    print(f"Initialization failed: {str(e)}")
    exit(1)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_question = data.get('question', '').strip()
        
        if not user_question:
            return jsonify({'answer': 'Please enter a valid question.'}), 400
        
        user_embedding = get_embeddings([user_question])
        similarities = torch.nn.functional.cosine_similarity(
            user_embedding, 
            faq_embeddings, 
            dim=1
        )

        print("\nCosine Similarities:")
        for idx, (sim, question) in enumerate(zip(similarities.tolist(), faq_data['question'].tolist())):
            print(f"{idx + 1:>2}. Similarity: {sim:.4f} | Question: {question}")
        
        best_match_idx = similarities.argmax().item()
        max_similarity = similarities[best_match_idx].item()
        answer = faq_data['answer'].iloc[best_match_idx]
        
        if max_similarity < MIN_SIMILARITY:
            return jsonify({
                'answer': "I'm not sure about that. Please contact admissions for more details.",
                'confidence': float(max_similarity)
            })
            
        return jsonify({
            'answer': answer,
            'confidence': float(max_similarity)
        })
        
    except Exception as e:
        return jsonify({'answer': 'An error occurred processing your request.'}), 500

# Static file serving
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=5000, debug=True)
