import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from step5_model_xai_dashboard.lstm_model import LSTMModel

def train_model(X_train, y_train, input_size, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

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

    torch.save(
        model.state_dict(),
        "step5_model_xai_dashboard/lstm_rul_model.pt"
    )

    return model
