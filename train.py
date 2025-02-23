import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import our improved classifier
from new import DeepReviewClassifier, clean_text


def load_and_preprocess_data(file_path):
    """Load and preprocess the Kaggle fake reviews dataset"""
    # Read the dataset
    df = pd.read_csv(file_path)

    # Extract features and labels
    texts = df['text_'].apply(clean_text).values  # Changed from 'text' to 'text_'
    labels = (df['label'] == 'OR').astype(int).values  # Changed from 'genuine' to 'OR'

    return texts, labels


def plot_training_metrics(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()


def train_model(file_path='fake_reviews_dataset.csv',
                model_save_path='best_model2.joblib',
                batch_size=32,
                epochs=3,
                learning_rate=2e-5,
                max_length=128,
                test_size=0.2,
                random_state=42):
    """Train the model on the Kaggle fake reviews dataset"""

    print("Loading and preprocessing data...")
    texts, labels = load_and_preprocess_data(file_path)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Initialize model
    print("Initializing model...")
    classifier = DeepReviewClassifier(max_length=max_length)

    # Training metrics history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Training loop
    print("Starting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training phase
        classifier.bert.train()
        classifier.classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        # Create batches
        for i in tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch + 1}/{epochs}'):
            batch_texts = X_train[i:i + batch_size]
            batch_labels = torch.tensor(y_train[i:i + batch_size], dtype=torch.long).to(device)

            # Tokenize and encode batch
            encoded = classifier.tokenizer(
                batch_texts.tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Forward pass
            outputs = classifier.forward(input_ids, attention_mask)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()
            classifier.optimizer.step()
            classifier.optimizer.zero_grad()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        # Calculate training metrics
        train_loss = total_loss / (len(X_train) / batch_size)
        train_acc = 100 * correct / total

        # Validation phase
        classifier.bert.eval()
        classifier.classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_texts = X_test[i:i + batch_size]
                batch_labels = torch.tensor(y_test[i:i + batch_size], dtype=torch.long).to(device)

                encoded = classifier.tokenizer(
                    batch_texts.tolist(),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )

                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                outputs = classifier.forward(input_ids, attention_mask)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        # Calculate validation metrics
        val_loss = val_loss / (len(X_test) / batch_size)
        val_acc = 100 * val_correct / val_total

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

    # Final evaluation
    print("\nGenerating final evaluation metrics...")
    y_pred = classifier.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot metrics
    plot_training_metrics(history)
    plot_confusion_matrix(y_test, y_pred)

    # Save the model
    print(f"\nSaving model to {model_save_path}...")
    import joblib
    joblib.dump(classifier, model_save_path)

    return classifier


if __name__ == "__main__":
    # Train the model
    classifier = train_model(
        file_path='uploads/fake_reviews_dataset.csv',
        model_save_path='best_model2.joblib',
        batch_size=32,
        epochs=3,
        learning_rate=2e-5,
        max_length=128
    )

