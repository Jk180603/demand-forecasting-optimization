import joblib
import numpy as np
import torch
import torch.nn as nn


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


def load_model():
    scaler = joblib.load("models/scaler.pkl")

    model = DemandForecastModel(input_dim=11)
    model.load_state_dict(torch.load("models/demand_model.pt", map_location="cpu"))
    model.eval()

    return model, scaler


def predict_demand(features):
    model, scaler = load_model()

    X = np.array([features])
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        prediction = model(torch.tensor(X_scaled, dtype=torch.float32)).item()

    return max(0, round(prediction))


def inventory_recommendation(predicted_demand, current_stock):
    if current_stock < predicted_demand:
        reorder_qty = predicted_demand - current_stock
        risk = "High Stockout Risk"
    elif current_stock < predicted_demand * 1.2:
        reorder_qty = round(predicted_demand * 0.3)
        risk = "Medium Stockout Risk"
    else:
        reorder_qty = 0
        risk = "Low Stockout Risk"

    return risk, reorder_qty