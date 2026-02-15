import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from MODELS.nlp_baseline import MaintenanceLogClassifier

def main():
    print("=== NLP Module Training Pipeline ===")
    
    # 1. Load Data
    data_path = os.path.join(ROOT_DIR, "DATA", "maintenance_logs.csv")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run DIGITAL_TWIN/logs_generator.py first.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Simple check
    print(f"Loaded {len(df)} logs.")
    
    # 2. Split Data
    X = df["log_text"]
    y = df["fault_type"]
    
    print("Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize and Train Model
    clf = MaintenanceLogClassifier()
    clf.train(X_train, y_train)
    
    # 4. Evaluate
    print("\n--- Evaluation on Test Set ---")
    metrics = clf.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['report'])
    
    # 5. Save Model
    model_path = os.path.join(ROOT_DIR, "MODELS", "nlp_baseline.pkl")
    clf.save(model_path)
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
