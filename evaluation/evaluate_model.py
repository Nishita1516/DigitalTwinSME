import os
import sys
import torch
import numpy as np

# Ensure project root is on the path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from DIGITAL_TWIN.data_loader import load_fd001
from DIGITAL_TWIN.preprocessing import add_rul, normalize_sensors
from DIGITAL_TWIN.twin_data_prep import split_engines, create_sliding_windows
from MODELS.predict_rul import load_model
from EVALUATION.metrics import compute_metrics, compute_classification_metrics
from DIGITAL_TWIN.config import SENSOR_COLS

if __name__ == "__main__":
    print("Loading dataset...")
    # Load and process data (much faster than training)
    data_path = os.path.join(
        ROOT_DIR, "DATA",
        "Sensor Data", "NASA C-MAPSS 1 Turbofan Engine Degradation Dataset", "train_FD001.txt"
    )
    fd001 = load_fd001(data_path)
    fd001 = add_rul(fd001)
    fd001, _ = normalize_sensors(fd001)
    
    # Split into train/val/test
    train_fd001, val_fd001, test_fd001 = split_engines(fd001)

    # Must match the checkpoint's 14 training features exactly.
    FEATURE_COLS = SENSOR_COLS
    WINDOW_SIZE = 30
    TARGET = "RUL"

    print("Preparing test data sequences...")
    # Create windows
    X_train, y_train = create_sliding_windows(train_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET)
    X_val, y_val = create_sliding_windows(val_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET)
    X_test, y_test = create_sliding_windows(test_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET)
    
    print("Loading pre-trained model...")
    # Load the already trained model
    model = load_model(input_size=len(SENSOR_COLS))
    
    print("Evaluating model...")
    # Run predictions on the test set
    y_true = y_test
    y_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().squeeze()

    mae, rmse = compute_metrics(y_true, y_pred)

    print("\n--- Regression Metrics ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    print("\n--- Classification Metrics (Threshold = 30) ---")
    accuracy, precision, recall, f1 = compute_classification_metrics(y_true, y_pred, threshold=30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
