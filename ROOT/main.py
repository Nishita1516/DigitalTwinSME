import os
import sys
import numpy as np
import torch


if __name__ == "__main__":
    # Ensure project root is on the path
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    from DIGITAL_TWIN.data_loader import load_fd001
    from DIGITAL_TWIN.preprocessing import add_rul, normalize_sensors
    from DIGITAL_TWIN.digital_twin import generate_digital_twins
    from DIGITAL_TWIN.twin_data_prep import (
        split_engines,
        create_sliding_windows,
        scale_windows,
    )

    from MODELS.train_lstm import train_model
    from MODELS.predict_rul import load_model, predict_rul
    from MODELS.shap_explainer import get_shap_explainer, explain_prediction
    from DASHBOARD.app import prepare_dashboard_payload
    from DIGITAL_TWIN.sequences import create_sequences

    from DIGITAL_TWIN.config import SENSOR_COLS, SEQ_LEN

    # STEP 1
    fd001 = load_fd001(
        "Sensor Data/NASA C-MAPSS 1 Turbofan Engine Degradation Dataset/train_FD001.txt"
    )

    # STEP 2
    fd001 = add_rul(fd001)
    fd001, _ = normalize_sensors(fd001)

    # STEP 3
    df_twin = generate_digital_twins(fd001)
    out_path = os.path.join(ROOT_DIR, "FD001_Synthetic_Digital_Twin_Dataset.csv")
    df_twin.to_csv(out_path, index=False)

    # STEP 4
    train_fd001, val_fd001, test_fd001 = split_engines(fd001)

    FEATURE_COLS = [c for c in fd001.columns if c.startswith("s")]
    WINDOW_SIZE = 30
    TARGET = "RUL"

    X_train, y_train = create_sliding_windows(
        train_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET
    )
    X_val, y_val = create_sliding_windows(
        val_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET
    )
    X_test, y_test = create_sliding_windows(
        test_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET
    )

    X_train, X_val, X_test = scale_windows(X_train, X_val, X_test)

    print("FD001 Digital Twin pipeline executed successfully")

    # DASHBOARD FINAL
    # 1. Create time-series sequences
    X, y = create_sequences(
        fd001,
        seq_length=SEQ_LEN,
        feature_cols=SENSOR_COLS,
        target_col="RUL",
    )

    # 2. Train LSTM model
    train_model(
        X_train=X,
        y_train=y,
        input_size=len(SENSOR_COLS),
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
    importance = importance.ravel()  # Flatten to 1D array

    # 6. Prepare dashboard data
    dashboard_output = prepare_dashboard_payload(
        engine_id=1,
        predicted_rul=predicted_rul,
        importance=importance.ravel(),
        feature_names=SENSOR_COLS,
    )

    assert importance.ndim == 1
    assert importance.shape[0] == len(SENSOR_COLS)

    from EVALUATION.metrics import compute_metrics
    from MODELS.predict_rul import evaluate_model

    # Here you should build a DataLoader over (X_test, y_test); placeholder:
    y_true, y_pred = y_test, model(
        torch.tensor(X_test, dtype=torch.float32)
    ).detach().numpy().squeeze()

    mae, rmse = compute_metrics(y_true, y_pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
