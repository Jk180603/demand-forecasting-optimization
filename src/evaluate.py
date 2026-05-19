import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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


X = np.load("models/X.npy")
y = np.load("models/y.npy")

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DemandForecastModel(input_dim=X.shape[1])
model.load_state_dict(
    torch.load("models/demand_model.pt", map_location="cpu")
)
model.eval()

with torch.no_grad():
    preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("Model Evaluation")
print("----------------")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test[:500], preds[:500], alpha=0.5)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Demand")
plt.tight_layout()
plt.savefig("reports/actual_vs_predicted.png")
plt.show()

with open("reports/evaluation_metrics.txt", "w") as f:
    f.write("Model Evaluation\n")
    f.write("----------------\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R²: {r2:.3f}\n")