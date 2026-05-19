import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("."))

from src.predict import predict_demand, inventory_recommendation


st.set_page_config(
    page_title="Demand Forecasting System",
    layout="wide"
)

st.title("Real-Time Demand Forecasting & Inventory Optimization")

st.write(
    "Predict future product demand and generate inventory recommendations using a PyTorch forecasting model."
)

st.sidebar.header("Input Features")

store = st.sidebar.number_input("Store ID", min_value=1, max_value=10, value=1)
item = st.sidebar.number_input("Item ID", min_value=1, max_value=50, value=1)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 2)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.number_input("Year", min_value=2013, max_value=2026, value=2017)
week_of_year = st.sidebar.slider("Week of Year", 1, 52, 25)

lag_1 = st.sidebar.number_input("Sales Lag 1", min_value=0, value=45)
lag_7 = st.sidebar.number_input("Sales Lag 7", min_value=0, value=50)
lag_30 = st.sidebar.number_input("Sales Lag 30", min_value=0, value=55)

rolling_mean_7 = st.sidebar.number_input("Rolling Mean 7", min_value=0.0, value=50.0)
rolling_mean_30 = st.sidebar.number_input("Rolling Mean 30", min_value=0.0, value=55.0)

current_stock = st.sidebar.number_input("Current Stock Level", min_value=0, value=40)

features = [
    store,
    item,
    day_of_week,
    month,
    year,
    week_of_year,
    lag_1,
    lag_7,
    lag_30,
    rolling_mean_7,
    rolling_mean_30,
]

if st.button("Predict Demand"):
    predicted_demand = predict_demand(features)

    risk, reorder_qty = inventory_recommendation(
        predicted_demand,
        current_stock
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Demand", predicted_demand)
    col2.metric("Current Stock", current_stock)
    col3.metric("Recommended Reorder Qty", reorder_qty)

    st.subheader("Inventory Risk")
    st.warning(risk)

    result_df = pd.DataFrame({
        "Metric": [
            "Predicted Demand",
            "Current Stock",
            "Recommended Reorder Quantity",
            "Risk Level"
        ],
        "Value": [
            predicted_demand,
            current_stock,
            reorder_qty,
            risk
        ]
    })

    st.dataframe(result_df, use_container_width=True)

    st.subheader("Business Interpretation")

    if reorder_qty > 0:
        st.write(
            "The model predicts demand higher than available stock. "
            "Reordering is recommended to reduce potential stockout risk."
        )
    else:
        st.write(
            "Current stock appears sufficient for the predicted demand window."
        )