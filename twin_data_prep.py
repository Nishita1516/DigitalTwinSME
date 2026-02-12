# STEP 4 – Engine-wise Train, Engine-safe splitting & windowing

# twin_data_prep.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Unique engines
def split_engines(fd001, seed=42):
    engines = fd001.engine_id.unique()
    np.random.seed(seed)

    # 70 / 15 / 15 split
    train_ids = np.random.choice(engines, int(0.7 * len(engines)), replace=False)
    remaining = list(set(engines) - set(train_ids))

    val_ids = np.random.choice(remaining, int(0.15 * len(engines)), replace=False)
    test_ids = list(set(remaining) - set(val_ids))

# Sliding Window Creation (Twin-Safe)

# Create splits
    train_fd001 = fd001[fd001.engine_id.isin(train_ids)]
    val_fd001   = fd001[fd001.engine_id.isin(val_ids)]
    test_fd001  = fd001[fd001.engine_id.isin(test_ids)]

    return train_fd001, val_fd001, test_fd001

def create_sliding_windows(fd001, window_size, feature_cols, target_col):
    X, y = [], []

    for engine_id in fd001.engine_id.unique():
        engine_data = fd001[fd001.engine_id == engine_id]
        features = engine_data[feature_cols].values
        target = engine_data[target_col].values

        for i in range(len(engine_data) - window_size):
            X.append(features[i:i + window_size])
            y.append(target[i + window_size])

    return np.array(X), np.array(y)

# Transform all splits using SAME scaler
def scale_windows(X_train, X_val, X_test):
    scaler = MinMaxScaler()

    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_2d)

    def scale(X):
        shape = X.shape
        X_scaled = scaler.transform(X.reshape(-1, shape[-1]))
        return X_scaled.reshape(shape)

    return scale(X_train), scale(X_val), scale(X_test)
