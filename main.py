# main.py
import numpy as np
if __name__ == "__main__": 
    from data_loader import load_fd001
    from preprocessing import add_rul, normalize_sensors
    from digital_twin import generate_digital_twins
    from twin_data_prep import split_engines, create_sliding_windows, scale_windows

    from step5_model_xai_dashboard.train_lstm import train_model
    from step5_model_xai_dashboard.predict_rul import load_model, predict_rul
    from step5_model_xai_dashboard.shap_explainer import (get_shap_explainer, explain_prediction)
    from dashboard_data import prepare_dashboard_payload
    from sequences import create_sequences

    from config import SENSOR_COLS, SEQ_LEN

# STEP 1
fd001 = load_fd001(
    "Sensor Data/NASA C-MAPSS 1 Turbofan Engine Degradation Dataset/train_FD001.txt")

# STEP 2
fd001 = add_rul(fd001)
fd001, _ = normalize_sensors(fd001)

# STEP 3
df_twin = generate_digital_twins(fd001)
df_twin.to_csv(
    r"c:\WORKFILES\dissertation\Digital Twin\FD001_Synthetic_Digital_Twin_Dataset.csv",
    index=False)

# STEP 4
train_fd001, val_fd001, test_fd001 = split_engines(fd001)

FEATURE_COLS = [c for c in fd001.columns if c.startswith("s")]
WINDOW_SIZE = 30
TARGET = "RUL"

X_train, y_train = create_sliding_windows(train_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET)
X_val, y_val     = create_sliding_windows(val_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET)
X_test, y_test   = create_sliding_windows(test_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET)

X_train, X_val, X_test = scale_windows(X_train, X_val, X_test)

print("FD001 Digital Twin pipeline executed successfully")
# print(fd001.head())
# To verify the failure cycle: expected output : 192
# print(fd001[fd001.engine_id == 1]["cycle"].max())

# DASHBOARD FINAL
# 1. Create time-series sequences
X, y = create_sequences(fd001,
    seq_length=30,
    feature_cols=SENSOR_COLS,
    target_col="RUL"
)

# 2. Train LSTM model
train_model(
    X_train=X,
    y_train=y,
    input_size=len(SENSOR_COLS)
)

# 3. Load trained model
model = load_model(input_size=len(SENSOR_COLS))

# 4. Predict RUL for one engine sample
predicted_rul = predict_rul(model, X[0])

# 5. Explain prediction using SHAP
explainer = get_shap_explainer(model, X)
shap_values = explain_prediction(explainer, X[0])
shap_array = np.array(shap_values)
importance = np.mean(np.abs(shap_array), axis=(0, 1))
importance = importance.ravel() # Flatten to 1D array

# 6. Prepare dashboard data
dashboard_output = prepare_dashboard_payload(
    engine_id=1,
    predicted_rul=predicted_rul,
    importance=importance.ravel(),
    feature_names=SENSOR_COLS
)
# print("STEP 5 OUTPUT:")

assert importance.ndim == 1
assert importance.shape[0] == len(SENSOR_COLS)

# print(dashboard_output)

# print("Available columns:", fd001.columns.tolist())
# print("Requested sensors:", SENSOR_COLS)

# print("Final importance shape:", importance.shape)
# print("Sensors:", len(SENSOR_COLS))

import data_loader
from evaluation.metrics import compute_metrics
from step5_model_xai_dashboard.predict_rul import evaluate_model

y_test, y_pred = evaluate_model(model, data_loader)
mae, rmse = compute_metrics(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
