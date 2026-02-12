import torch
from step5_model_xai_dashboard.lstm_model import LSTMModel


def load_model(input_size):
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(
        torch.load("step5_model_xai_dashboard/lstm_rul_model.pt")
    )
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
