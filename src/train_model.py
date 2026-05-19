import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import mlflow
import mlflow.pytorch


class DemandDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class DemandForecastModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


def train():
    X = np.load("models/X.npy")
    y = np.load("models/y.npy")

    dataset = DemandDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)

    model = DemandForecastModel(input_dim=X.shape[1])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    mlflow.set_experiment("demand-forecasting")

    with mlflow.start_run():
        mlflow.log_param("model", "feedforward_pytorch")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("batch_size", 256)

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                preds = model(batch_X)
                loss = loss_fn(preds, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    preds = model(batch_X)
                    loss = loss_fn(preds, batch_y)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}"
            )

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        torch.save(model.state_dict(), "models/demand_model.pt")

        mlflow.pytorch.log_model(model, "model")

        print("Model training completed")
        print("Model saved to models/demand_model.pt")


if __name__ == "__main__":
    train()