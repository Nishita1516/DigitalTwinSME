"""
Train a reproducible Random Forest baseline using the same engine split
and StandardScaler as the clean LSTM experiment.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from DIGITAL_TWIN.config import RUL_FEATURE_COLS, SEQ_LEN
from DIGITAL_TWIN.data_loader import load_fd001
from DIGITAL_TWIN.preprocessing import add_rul
from DIGITAL_TWIN.twin_data_prep import create_sliding_windows
from EVALUATION.metrics import compute_metrics, compute_classification_metrics


def transform_with_saved_scaler(df, scaler):
    """Apply the saved StandardScaler."""
    df = df.copy()
    df[RUL_FEATURE_COLS] = scaler.transform(df[RUL_FEATURE_COLS].astype(float))
    return df


def aggregate_windows(X):
    """Aggregate each window into mean, std, min and max features."""
    return np.hstack([X.mean(1), X.std(1), X.min(1), X.max(1)])


def metric_record(y_true, y_pred):
    mae, rmse = compute_metrics(y_true, y_pred)
    acc, prec, rec, f1 = compute_classification_metrics(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "rul_le_30_accuracy": float(acc),
        "rul_le_30_precision": float(prec),
        "rul_le_30_recall": float(rec),
        "rul_le_30_f1": float(f1),
        "samples": int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="rf_fd001_clean_seed42_run1")
    parser.add_argument("--lstm-experiment", default="fd001_clean_seed42_run1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    # Load LSTM experiment
    lstm_dir = ROOT_DIR / "MODELS" / "experiments" / args.lstm_experiment
    metadata = json.loads((lstm_dir / "results.json").read_text(encoding="utf-8"))
    scaler = joblib.load(lstm_dir / "sensor_standard_scaler.joblib")

    # Create experiment directory
    experiment_dir = ROOT_DIR / "MODELS" / "experiments" / args.experiment_name
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(f"{experiment_dir} already exists.")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data_path = ROOT_DIR / "DATA" / "Sensor Data" / "NASA C-MAPSS 1 Turbofan Engine Degradation Dataset" / "train_FD001.txt"
    fd001 = add_rul(load_fd001(str(data_path)))

    train_ids = metadata["train_engine_ids"]
    val_ids = metadata["validation_engine_ids"]
    test_ids = metadata["test_engine_ids"]

    train_df = fd001[fd001.engine_id.isin(train_ids)]
    val_df = fd001[fd001.engine_id.isin(val_ids)]
    test_df = fd001[fd001.engine_id.isin(test_ids)]

    # Apply preprocessing
    train_df, val_df, test_df = [
        transform_with_saved_scaler(df, scaler)
        for df in (train_df, val_df, test_df)
    ]

    # Create windows
    X_train, y_train = create_sliding_windows(train_df, SEQ_LEN, RUL_FEATURE_COLS, "RUL")
    X_val, y_val = create_sliding_windows(val_df, SEQ_LEN, RUL_FEATURE_COLS, "RUL")
    X_test, y_test = create_sliding_windows(test_df, SEQ_LEN, RUL_FEATURE_COLS, "RUL")

    # Aggregate features
    X_train, X_val, X_test = map(aggregate_windows, (X_train, X_val, X_test))

    # Train model
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    val_pred, test_pred = rf.predict(X_val), rf.predict(X_test)
    val_metrics = metric_record(y_val, val_pred)
    test_metrics = metric_record(y_test, test_pred)

    # Save artifacts
    joblib.dump(rf, experiment_dir / "random_forest.joblib")

    for split, y_true, y_pred in (
        ("validation", y_val, val_pred),
        ("test", y_test, test_pred),
    ):
        pd.DataFrame({"true_rul": y_true, "predicted_rul": y_pred}).to_csv(
            experiment_dir / f"{split}_predictions.csv", index=False
        )

    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "Random Forest",
        "dataset": metadata["dataset"],
        "features": metadata["features"],
        "sequence_length": metadata["sequence_length"],
        "split": metadata["split"],
        "preprocessing": metadata["preprocessing"],
        "aggregation": ["mean", "std", "min", "max"],
        "feature_count": int(X_train.shape[1]),
        "model": {
            "type": "RandomForestRegressor",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.seed,
        },
        "train_engine_ids": train_ids,
        "validation_engine_ids": val_ids,
        "test_engine_ids": test_ids,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    (experiment_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    for name, metrics in (("Validation", val_metrics), ("Test", test_metrics)):
        print(f"\\n{name} metrics")
        print(json.dumps(metrics, indent=2))

    print(f"\\nExperiment saved to\\n{experiment_dir}")


if __name__ == "__main__":
    main()