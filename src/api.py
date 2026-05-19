from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_demand, inventory_recommendation


app = FastAPI(
    title="Demand Forecasting API",
    description="PyTorch-based demand forecasting and inventory optimization API",
    version="1.0.0",
)


class DemandRequest(BaseModel):
    store: int
    item: int
    day_of_week: int
    month: int
    year: int
    week_of_year: int
    lag_1: float
    lag_7: float
    lag_30: float
    rolling_mean_7: float
    rolling_mean_30: float
    current_stock: int


@app.get("/")
def root():
    return {
        "message": "Demand Forecasting API is running"
    }


@app.post("/predict")
def predict(request: DemandRequest):
    features = [
        request.store,
        request.item,
        request.day_of_week,
        request.month,
        request.year,
        request.week_of_year,
        request.lag_1,
        request.lag_7,
        request.lag_30,
        request.rolling_mean_7,
        request.rolling_mean_30,
    ]

    predicted_demand = predict_demand(features)

    risk, reorder_qty = inventory_recommendation(
        predicted_demand,
        request.current_stock,
    )

    return {
        "predicted_demand": predicted_demand,
        "current_stock": request.current_stock,
        "stockout_risk": risk,
        "recommended_reorder_quantity": reorder_qty,
    }