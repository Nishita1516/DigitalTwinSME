from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def compute_classification_metrics(y_true, y_pred, threshold=30):
    # Convert continuous RUL to binary classes (1: failure within threshold, 0: safe)
    y_true_binary = (np.array(y_true) <= threshold).astype(int)
    y_pred_binary = (np.array(y_pred) <= threshold).astype(int)
    
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return accuracy, precision, recall, f1
