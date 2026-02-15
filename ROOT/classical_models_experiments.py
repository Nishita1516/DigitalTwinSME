import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Ensure project root is on the path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from DIGITAL_TWIN.data_loader import load_fd001
from DIGITAL_TWIN.preprocessing import add_rul, normalize_sensors
from DIGITAL_TWIN.twin_data_prep import split_engines, create_sliding_windows, scale_windows
from DIGITAL_TWIN.config import SENSOR_COLS, SEQ_LEN
from EVALUATION.metrics import compute_metrics


def aggregate_windows(X: np.ndarray) -> np.ndarray:
    """
    Aggregate a 3D sliding-window tensor (samples, window_size, features)
    into 2D features using simple statistics over the time dimension.
    """
    # X shape: (n_samples, window_size, n_features)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    min_ = X.min(axis=1)
    max_ = X.max(axis=1)

    # Concatenate along feature axis → (n_samples, 4 * n_features)
    return np.concatenate([mean, std, min_, max_], axis=1)


def prepare_data():
    """
    Load FD001, compute RUL, normalize sensors, create sliding windows,
    scale them and aggregate into classical ML features.
    """
    # 1) Load and preprocess FD001
    fd001 = load_fd001(
        os.path.join(
            ROOT_DIR,
            "Sensor Data",
            "NASA C-MAPSS 1 Turbofan Engine Degradation Dataset",
            "train_FD001.txt",
        )
    )
    fd001 = add_rul(fd001)
    fd001, _ = normalize_sensors(fd001)

    # 2) Engine-wise splits (train / val / test)
    train_fd001, val_fd001, test_fd001 = split_engines(fd001)

    # 3) Sliding windows (same window length and sensor subset as LSTM)
    feature_cols = SENSOR_COLS
    target_col = "RUL"
    window_size = SEQ_LEN

    X_train_w, y_train = create_sliding_windows(
        train_fd001, window_size, feature_cols, target_col
    )
    X_val_w, y_val = create_sliding_windows(
        val_fd001, window_size, feature_cols, target_col
    )
    X_test_w, y_test = create_sliding_windows(
        test_fd001, window_size, feature_cols, target_col
    )

    # 4) Scale windows using a shared scaler fit on train only
    X_train_w, X_val_w, X_test_w = scale_windows(X_train_w, X_val_w, X_test_w)

    # 5) Aggregate windows into classical ML features
    X_train = aggregate_windows(X_train_w)
    X_val = aggregate_windows(X_val_w)
    X_test = aggregate_windows(X_test_w)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    results = []

    for split_name, X_split, y_split in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = rf.predict(X_split)
        mae, rmse = compute_metrics(y_split, y_pred)
        results.append(
            {
                "model": "RandomForestRegressor",
                "split": split_name,
                "mae": mae,
                "rmse": rmse,
            }
        )

    return rf, results


def run_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test):
    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_train, y_train)

    results = []

    for split_name, X_split, y_split in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = gb.predict(X_split)
        mae, rmse = compute_metrics(y_split, y_pred)
        results.append(
            {
                "model": "GradientBoostingRegressor",
                "split": split_name,
                "mae": mae,
                "rmse": rmse,
            }
        )

    return gb, results


def main():
    # 1) Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    # 2) Train and evaluate classical models
    all_results = []

    rf_model, rf_results = run_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    all_results.extend(rf_results)

    gb_model, gb_results = run_gradient_boosting(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    all_results.extend(gb_results)

    results_df = pd.DataFrame(all_results)

    # 3) Save results to CSV for Chapter 5 tables
    results_path = os.path.join(ROOT_DIR, "EVALUATION", "classical_rul_results.csv")
    results_df.to_csv(results_path, index=False)

    print("Classical model RUL results:")
    print(results_df)
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()

