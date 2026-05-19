import numpy as np
import pandas as pd


np.random.seed(42)

dates = pd.date_range(start="2022-01-01", periods=730, freq="D")

products = ["Product_A", "Product_B", "Product_C"]
stores = ["Store_1", "Store_2"]

rows = []

for product in products:
    for store in stores:
        base_demand = np.random.randint(40, 120)

        for i, date in enumerate(dates):
            weekday_effect = 15 if date.weekday() in [5, 6] else 0
            yearly_seasonality = 20 * np.sin(2 * np.pi * i / 365)
            promotion = np.random.choice([0, 1], p=[0.85, 0.15])
            promo_effect = promotion * np.random.randint(20, 60)
            noise = np.random.normal(0, 10)

            units_sold = max(
                0,
                base_demand
                + weekday_effect
                + yearly_seasonality
                + promo_effect
                + noise,
            )

            price = np.random.uniform(8, 30)
            stock_level = np.random.randint(20, 250)

            rows.append(
                {
                    "date": date,
                    "product_id": product,
                    "store_id": store,
                    "price": round(price, 2),
                    "promotion": promotion,
                    "stock_level": stock_level,
                    "units_sold": round(units_sold),
                }
            )

df = pd.DataFrame(rows)

df.to_csv("data/retail_demand.csv", index=False)

print("Dataset created successfully")
print(df.head())
print(df.shape)