import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os

class MaintenanceLogClassifier:
    def __init__(self):
        """
        Initialize a pipeline with TF-IDF and Logistic Regression.
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('clf', LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000))
        ])
        
    def train(self, texts, labels):
        """
        Train the classifier.
        
        Args:
            texts: List or Series of text data
            labels: List or Series of target labels (fault types)
        """
        print(f"Training on {len(texts)} samples...")
        self.pipeline.fit(texts, labels)
        print("Training complete.")
        
    def predict(self, texts):
        """
        Predict labels for new texts.
        """
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts):
        """
        Predict class probabilities.
        """
        return self.pipeline.predict_proba(texts)
    
    def evaluate(self, texts, labels):
        """
        Evaluate the model and return metrics.
        """
        preds = self.predict(texts)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds)
        
        return {
            "accuracy": acc,
            "report": report
        }
        
    def save(self, path):
        """
        Save the trained pipeline to disk.
        """
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        """
        Load a trained pipeline from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        self.pipeline = joblib.load(path)
        print(f"Model loaded from {path}")

# Example usage if run directly
if __name__ == "__main__":
    pass
