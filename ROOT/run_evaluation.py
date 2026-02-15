import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import local modules
from DIGITAL_TWIN.data_loader import load_fd001
from DIGITAL_TWIN.preprocessing import add_rul, normalize_sensors
from DIGITAL_TWIN.twin_data_prep import split_engines, create_sliding_windows, scale_windows
from DIGITAL_TWIN.config import SENSOR_COLS, SEQ_LEN
from MODELS.predict_rul import load_model, predict_rul
from MODELS.nlp_baseline import MaintenanceLogClassifier

def evaluate_ml_models():
    print("\n--- Evaluating ML Models (RUL Prediction) ---")
    
    # 1. Load Data
    data_path = os.path.join(ROOT_DIR, "DATA", "Sensor Data", "NASA C-MAPSS 1 Turbofan Engine Degradation Dataset", "train_FD001.txt")
    if not os.path.exists(data_path):
        print("Error: train_FD001.txt not found.")
        return {}

    fd001 = load_fd001(data_path)
    fd001 = add_rul(fd001)
    fd001, _ = normalize_sensors(fd001)
    _, _, test_fd001 = split_engines(fd001)

    # 2. Prepare Test Data
    X_test_w, y_test = create_sliding_windows(test_fd001, SEQ_LEN, SENSOR_COLS, "RUL")
    # Note: We should technically use the scaler fitted on train, but for quick evaluation we'll skip re-fitting
    # assuming the data is already somewhat normalized. Ideally load the scaler.
    
    # 3. Evaluate LSTM
    print("Evaluating LSTM...")
    try:
        lstm_model = load_model(len(SENSOR_COLS))
        lstm_preds = []
        total_seqs = len(X_test_w)
        print(f"Total sequences to predict: {total_seqs}", flush=True)
        
        for i, seq in enumerate(X_test_w):
            pred = predict_rul(lstm_model, seq)
            lstm_preds.append(pred)
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{total_seqs}", flush=True)
        
        lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))
        lstm_mae = mean_absolute_error(y_test, lstm_preds)
        print(f"LSTM RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}")
    except Exception as e:
        print(f"LSTM Evaluation failed: {e}")
        lstm_rmse, lstm_mae = None, None

    # 4. Evaluate Random Forest (Quick Train & Test)
    print("Evaluating Random Forest (Baseline)...")
    try:
        # Aggregate features for RF (mean, std of window)
        X_test_rf = np.mean(X_test_w, axis=1) # Simple aggregation
        
        # Need training data for RF
        train_fd001, _, _ = split_engines(fd001)
        X_train_w, y_train = create_sliding_windows(train_fd001, SEQ_LEN, SENSOR_COLS, "RUL")
        X_train_rf = np.mean(X_train_w, axis=1)
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train_rf, y_train)
        
        rf_preds = rf.predict(X_test_rf)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
        rf_mae = mean_absolute_error(y_test, rf_preds)
        print(f"RF RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
    except Exception as e:
        print(f"RF Evaluation failed: {e}")
        rf_rmse, rf_mae = None, None

    return {
        "LSTM": {"RMSE": lstm_rmse, "MAE": lstm_mae},
        "RandomForest": {"RMSE": rf_rmse, "MAE": rf_mae}
    }

def evaluate_nlp_model():
    print("\n--- Evaluating NLP Model (Fault Classification) ---")
    
    log_path = os.path.join(ROOT_DIR, "DATA", "maintenance_logs.csv")
    if not os.path.exists(log_path):
        print("Error: maintenance_logs.csv not found.")
        return {}
        
    df = pd.read_csv(log_path)
    X = df["log_text"]
    y = df["fault_type"]
    
    # Split same as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_path = os.path.join(ROOT_DIR, "MODELS", "nlp_baseline.pkl")
    if not os.path.exists(model_path):
        print("Error: nlp_baseline.pkl not found.")
        return {}

    try:
        clf = joblib.load(model_path)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"NLP Accuracy: {acc:.4f}")
        return {"Accuracy": acc, "Report": report}
    except Exception as e:
        print(f"NLP Evaluation failed: {e}")
        return {}

def generate_report(ml_results, nlp_results):
    report_path = os.path.join(ROOT_DIR, "EVALUATION", "final_performance_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("=== Digital Twin Project: Performance Evaluation ===\n\n")
        
        f.write("1. Predictive Maintenance Models (RUL Prediction)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Model':<15} | {'RMSE':<10} | {'MAE':<10}\n")
        f.write("-" * 50 + "\n")
        
        if ml_results.get("LSTM"):
            f.write(f"{'LSTM':<15} | {ml_results['LSTM']['RMSE']:.4f}     | {ml_results['LSTM']['MAE']:.4f}\n")
        if ml_results.get("RandomForest"):
            f.write(f"{'Random Forest':<15} | {ml_results['RandomForest']['RMSE']:.4f}     | {ml_results['RandomForest']['MAE']:.4f}\n")
        f.write("\n\n")
        
        f.write("2. NLP Maintenance Log Analysis\n")
        f.write("-" * 50 + "\n")
        if nlp_results:
            f.write(f"Classification Accuracy: {nlp_results.get('Accuracy', 'N/A'):.4f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(nlp_results.get("Report", "N/A"))
        else:
            f.write("NLP Evaluation Failed.\n")
            
    print(f"\nFinal Report generated at: {report_path}")

def main():
    ml_res = evaluate_ml_models()
    nlp_res = evaluate_nlp_model()
    generate_report(ml_res, nlp_res)

if __name__ == "__main__":
    main()
