import os

import torch

from MODELS.lstm_model import LSTMModel
from DIGITAL_TWIN.config import LSTM_MODEL_FILENAME


def load_model(input_size, model_filename=LSTM_MODEL_FILENAME):
    model = LSTMModel(input_size=input_size)
    models_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM checkpoint not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_rul(model, sequence):
    with torch.no_grad():
        sequence = torch.tensor(
            sequence, dtype=torch.float32
        ).unsqueeze(0)
        prediction = model(sequence)
        return float(prediction.item())


import numpy as np
from torch.utils.data import DataLoader

# create a function to evaluate the model
def evaluate_model(model, test_loader):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    return y_true, y_pred
