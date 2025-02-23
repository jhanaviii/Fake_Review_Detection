import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import joblib
import re
from datetime import datetime
from flask_cors import CORS
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Import the deep learning classifier
from new import DeepReviewClassifier, clean_text


class ReviewAnalysisApp:
    def __init__(self, model_path='best_model2.joblib'):
        # Initialize Flask app
        self.app = Flask(__name__, static_folder='static')
        CORS(self.app)

        # Setup routes
        self.setup_routes()

        # Load pre-trained model
        self.load_model(model_path)

        # Initialize review database
        self.reviews_db = []

    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            self.model = DeepReviewClassifier.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return send_from_directory('static', 'index.html')

        @self.app.route('/analyze', methods=['POST'])
        def analyze_review():
            try:
                data = request.json
                review_text = data.get('review', '')

                if not review_text:
                    return jsonify({'error': 'No review text provided'}), 400

                # Preprocess and analyze review
                cleaned_text = clean_text(review_text)
                prediction = self.model.predict([cleaned_text])[0]

                sentiment = 'Genuine' if prediction == 1 else 'Fake'
                analysis = self.advanced_review_analysis(review_text)

                # Get confidence scores
                proba = self.model.predict_proba([cleaned_text])[0]
                confidence = float(max(proba))

                review_entry = {
                    'text': review_text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'word_count': analysis['word_count']
                }
                self.reviews_db.append(review_entry)

                return jsonify(review_entry)
            except Exception as e:
                print(f"Error in analyze_review: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/bulk_upload', methods=['POST'])
        def bulk_upload():
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file uploaded'}), 400

                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No selected file'}), 400

                filename = secure_filename(file.filename)
                filepath = os.path.join('uploads', filename)
                os.makedirs('uploads', exist_ok=True)
                file.save(filepath)

                results = self.analyze_review_file(filepath)

                os.remove(filepath)

                return jsonify(results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/reviews', methods=['GET'])
        def get_reviews():
            try:
                return jsonify(sorted(
                    self.reviews_db,
                    key=lambda x: x['timestamp'],
                    reverse=True
                ))
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/metrics')
        def get_metrics():
            # Load the classification report and confusion matrix
            with open('classification_report.txt', 'r') as f:
                classification_report = f.read()
            return jsonify(classification_report)

        @self.app.route('/feature_importance', methods=['GET'])
        def get_feature_importance():
            try:
                # Example feature importance (replace with actual data from your model)
                feature_importance = [0.1, 0.3, 0.5, 0.2, 0.4]
                return jsonify(feature_importance)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/class_distribution', methods=['GET'])
        def get_class_distribution():
            try:
                genuine_count = len([review for review in self.reviews_db if review['sentiment'] == 'Genuine'])
                fake_count = len([review for review in self.reviews_db if review['sentiment'] == 'Fake'])
                return jsonify({
                    'genuine': genuine_count,
                    'fake': fake_count
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def advanced_review_analysis(self, review_text):
        return {
            'word_count': len(review_text.split()),
            'contains_emoji': bool(re.search(r'[^\w\s,.]', review_text)),
            'avg_word_length': np.mean([len(word) for word in review_text.split()]),
            'exclamation_count': review_text.count('!'),
            'question_count': review_text.count('?')
        }

    def analyze_review_file(self, filepath):
        try:
            df = pd.read_csv(filepath)
            results = []

            for _, row in df.iterrows():
                review_text = str(row.iloc[0])
                cleaned_text = clean_text(review_text)
                prediction = self.model.predict([cleaned_text])[0]
                proba = self.model.predict_proba([cleaned_text])[0]

                sentiment = 'Genuine' if prediction == 1 else 'Fake'
                confidence = float(max(proba))
                analysis = self.advanced_review_analysis(review_text)

                review_entry = {
                    'text': review_text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'word_count': analysis['word_count']
                }
                results.append(review_entry)
                self.reviews_db.append(review_entry)

            return results
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    def run(self, debug=True, port=5000):
        os.makedirs('static', exist_ok=True)
        self.app.run(debug=debug, port=port)




if __name__ == "__main__":
    app = ReviewAnalysisApp()
    app.run()

