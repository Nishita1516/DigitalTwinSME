# Time-Series Windows
# Converts raw cycles → LSTM-ready sequences
import numpy as np

from config import SENSOR_COLS

def create_sequences(df, seq_length, feature_cols, target_col):
    """
    Converts sensor data into sliding window sequences for LSTM.
    """
    X, y = [], []

    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        engine_data = engine_data.sort_values('cycle')

        features = engine_data[feature_cols].values
        target = engine_data[target_col].values

        for i in range(len(engine_data) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length])

    return np.array(X), np.array(y)

   