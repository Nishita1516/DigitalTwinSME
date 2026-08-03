import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from MODELS.lstm_model import LSTMModel

def train_model(X_train, y_train, input_size, epochs=30, model_path=None, seed=42):
    """Train on StandardScaler-normalised SENSOR_COLS windows.

    ``model_path`` is deliberately required to persist a new experiment. Keep
    its checkpoint separate from ``MODELS/lstm_model.pt`` unless it has passed
    the recorded evaluation protocol.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    # Saving is explicit: evaluation/dashboard execution must never replace a
    # published checkpoint as a side effect.
    if model_path is not None:
        torch.save(model.state_dict(), model_path)

    return model
