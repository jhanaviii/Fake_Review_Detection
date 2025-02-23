import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import re


class DeepReviewClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name='distilbert-base-uncased', max_length=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.max_length = max_length

        # Initialize tokenizer and base model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

        # Move models to device
        self.bert = self.bert.to(self.device)
        self.classifier = self.classifier.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.bert.parameters()) + list(self.classifier.parameters()),
            lr=2e-5
        )

    def predict_proba(self, texts):
        """Predict class probabilities for the provided texts."""
        self.bert.eval()
        self.classifier.eval()

        probas = []
        with torch.no_grad():
            for text in texts:
                encoded = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                outputs = self.forward(input_ids, attention_mask)
                probas.append(torch.softmax(outputs, dim=1).cpu().numpy()[0])

        return np.array(probas)

    def predict(self, texts):
        """Predict class labels for the provided texts."""
        if isinstance(texts, str):
            texts = [texts]
        probs = self.predict_proba(texts)
        return (probs[:, 1] >= 0.5).astype(int)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

    def save(self, path):
        """Save the model using joblib."""
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        """Load the model from a file."""
        return joblib.load(path)


def clean_text(text):
    """Clean the input text by removing special characters and converting to lowercase."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text.strip()


# Alias for compatibility with existing code
AmazonReviewClassifier = DeepReviewClassifier