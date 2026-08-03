"""Train one reproducible, engine-held-out FD001 LSTM experiment.

This command never writes to MODELS/lstm_model.pt or MODELS/lstm_rul_model.pt.
Each experiment receives its own directory containing the checkpoint, fitted
training scaler, split engine IDs, configuration, and validation/test metrics.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from DIGITAL_TWIN.config import RUL_FEATURE_COLS, SEQ_LEN
from DIGITAL_TWIN.data_loader import load_fd001
from DIGITAL_TWIN.preprocessing import add_rul
from DIGITAL_TWIN.twin_data_prep import create_sliding_windows, split_engines
from EVALUATION.metrics import compute_metrics, compute_classification_metrics
from MODELS.train_lstm import train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def standardise_from_training(train_df, *other_dfs):
    """Fit one scaler on training engines only, then transform every split."""
    scaler = StandardScaler()
    train_df = train_df.copy()
    scaler.fit(train_df[RUL_FEATURE_COLS])
    train_df.loc[:, RUL_FEATURE_COLS] = scaler.transform(train_df[RUL_FEATURE_COLS])

    transformed = []
    for df in other_dfs:
        transformed_df = df.copy()
        transformed_df.loc[:, RUL_FEATURE_COLS] = scaler.transform(
            transformed_df[RUL_FEATURE_COLS]
        )
        transformed.append(transformed_df)
    return (train_df, *transformed), scaler


def predict(model, X: np.ndarray) -> np.ndarray:
    model = model.cpu().eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()


def metric_record(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae, rmse = compute_metrics(y_true, y_pred)
    accuracy, precision, recall, f1 = compute_classification_metrics(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "rul_le_30_accuracy": float(accuracy),
        "rul_le_30_precision": float(precision),
        "rul_le_30_recall": float(recall),
        "rul_le_30_f1": float(f1),
        "samples": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="fd001_clean_seed42")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    experiment_dir = ROOT_DIR / "MODELS" / "experiments" / args.experiment_name
    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}. "
            "Choose a new --experiment-name; this script never overwrites artifacts."
        )
    experiment_dir.mkdir(parents=True)

    set_seed(args.seed)
    data_path = ROOT_DIR / "DATA" / "Sensor Data" / (
        "NASA C-MAPSS 1 Turbofan Engine Degradation Dataset"
    ) / "train_FD001.txt"

    print("Loading FD001 and computing RUL labels...", flush=True)
    fd001 = add_rul(load_fd001(str(data_path)))
    train_df, val_df, test_df = split_engines(fd001, seed=args.seed)
    train_df, val_df, test_df, scaler = standardise_from_training(
        train_df, val_df, test_df
    )

    X_train, y_train = create_sliding_windows(train_df, SEQ_LEN, RUL_FEATURE_COLS, "RUL")
    X_val, y_val = create_sliding_windows(val_df, SEQ_LEN, RUL_FEATURE_COLS, "RUL")
    X_test, y_test = create_sliding_windows(test_df, SEQ_LEN, RUL_FEATURE_COLS, "RUL")
    print(
        f"Windows: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}",
        flush=True,
    )

    checkpoint_path = experiment_dir / "lstm_state_dict.pt"
    print(f"Training {args.epochs} epochs; saving only to {checkpoint_path}", flush=True)
    model = train_model(
        X_train, y_train, input_size=len(RUL_FEATURE_COLS), epochs=args.epochs,
        model_path=str(checkpoint_path), seed=args.seed,
    )

    val_metrics = metric_record(y_val, predict(model, X_val))
    test_metrics = metric_record(y_test, predict(model, X_test))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_path.relative_to(ROOT_DIR)),
        "features": RUL_FEATURE_COLS,
        "sequence_length": SEQ_LEN,
        "split": {"train": 70, "validation": 15, "test": 15, "seed": args.seed},
        "preprocessing": "StandardScaler fitted on training-engine rows only",
        "model": {"type": "LSTMModel", "hidden_size": 64, "num_layers": 2},
        "epochs": args.epochs,
        "train_engine_ids": sorted(int(x) for x in train_df.engine_id.unique()),
        "validation_engine_ids": sorted(int(x) for x in val_df.engine_id.unique()),
        "test_engine_ids": sorted(int(x) for x in test_df.engine_id.unique()),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(scaler, experiment_dir / "sensor_standard_scaler.joblib")
    (experiment_dir / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n--- Validation metrics ---")
    print(json.dumps(val_metrics, indent=2))
    print("\n--- Held-out test metrics ---")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved reproducible experiment: {experiment_dir}")


if __name__ == "__main__":
    main()
