# Time-Series Windows
# Converts raw cycles → LSTM-ready sequences
from DIGITAL_TWIN.twin_data_prep import create_sliding_windows

def create_sequences(df, seq_length, feature_cols, target_col):
    """
    Converts sensor data into sliding window sequences for LSTM.
    """
    # Retain the public function while delegating to the one window builder.
    return create_sliding_windows(df, seq_length, feature_cols, target_col)
