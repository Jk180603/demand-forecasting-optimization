import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv("data/train.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort values
df = df.sort_values(["store", "item", "date"])

# Time-based features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

# Lag features
df["lag_1"] = (
    df.groupby(["store", "item"])["sales"]
    .shift(1)
)

df["lag_7"] = (
    df.groupby(["store", "item"])["sales"]
    .shift(7)
)

df["lag_30"] = (
    df.groupby(["store", "item"])["sales"]
    .shift(30)
)

# Rolling mean features
df["rolling_mean_7"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

df["rolling_mean_30"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(lambda x: x.shift(1).rolling(30).mean())
)

# Drop missing rows created by lagging
df = df.dropna().reset_index(drop=True)

# Features and target
feature_cols = [
    "store",
    "item",
    "day_of_week",
    "month",
    "year",
    "week_of_year",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30"
]

target_col = "sales"

X = df[feature_cols]
y = df[target_col]

# Scale features
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save processed arrays
np.save("models/X.npy", X_scaled)
np.save("models/y.npy", y.values)

# Save cleaned dataframe
df.to_csv("data/processed_data.csv", index=False)

print("Preprocessing completed")
print("Processed shape:", X_scaled.shape)