import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from new import DeepReviewClassifier, clean_text

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Generate and save confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Genuine'],
                yticklabels=['Fake', 'Genuine'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return cm

def evaluate_model(data_path='fake_reviews_dataset.csv',
                   model_path='best_model2.joblib',
                   test_size=0.2,
                   random_state=42):
    """Evaluate the trained model and generate all metrics"""

    print("Loading data...")
    df = pd.read_csv(data_path)

    # Standardize column names to avoid KeyError issues
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Ensure consistent label mapping with 'OR' for genuine
    texts = df['text_'].apply(clean_text).values
    labels = (df['label'] == 'OR').astype(int).values

    # Split data
    _, X_test, _, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    print("Loading model...")
    model = joblib.load(model_path)

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    # Debug: Ensure label consistency
    print("Unique labels in y_test: ", np.unique(y_test))
    print("Unique predictions: ", np.unique(y_pred))

    # Generate classification report
    report = classification_report(y_test, y_pred)
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    print("\nClassification Report:")
    print(report)

    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate additional metrics
    probas = model.predict_proba(X_test)
    confidence_scores = np.max(probas, axis=1)

    print("\nAdditional Metrics:")
    print(f"Average Confidence Score: {confidence_scores.mean():.4f}")
    print(f"Median Confidence Score: {np.median(confidence_scores):.4f}")

    # Save detailed results
    results = pd.DataFrame({
        'Text': X_test,
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Confidence': confidence_scores
    })
    results.to_csv('evaluation_results.csv', index=False)

    print("\nFiles generated:")
    print("1. confusion_matrix.png - Visual confusion matrix")
    print("2. classification_report.txt - Detailed classification metrics")
    print("3. evaluation_results.csv - Detailed predictions with confidence scores")

if __name__ == "__main__":
    evaluate_model()
