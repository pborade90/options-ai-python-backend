import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from sklearn.preprocessing import MinMaxScaler
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

# ✅ Generate Unique ID
def generate_unique_id():
    return str(uuid.uuid4())

# ✅ Fetch stock data
def get_stock_data(stock_symbol: str, years: int = 20):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=years)
    stock = yf.Ticker(stock_symbol)
    data = stock.history(start=start_date, end=end_date, interval="1d")
    if not data.empty:
        data.reset_index(inplace=True)
        return data
    return None

# ✅ Train LSTM Model
def train_lstm(stock_symbol):
    data = get_stock_data(stock_symbol)
    if data is None:
        return None

    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Scaled_Close'] = scaler.fit_transform(data[['Close']])

    x_train, y_train = [], []
    seq_length = 100
    for i in range(seq_length, len(data['Scaled_Close']) - 1):
        x_train.append(data['Scaled_Close'][i-seq_length:i])
        y_train.append(data['Scaled_Close'][i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=16, epochs=2, verbose=1)
    model.save(f"{stock_symbol}_model.h5")

    return model

# ✅ Predict Stock Movement with Unique ID for Each Date
def predict_stock(stock_symbol):
    model_path = f"{stock_symbol}_model.h5"

    if not os.path.exists(model_path):
        model = train_lstm(stock_symbol)
        if model is None:
            return {"error": "Stock data not found! Unable to train model."}
    else:
        model = tf.keras.models.load_model(model_path)

    data = get_stock_data(stock_symbol)
    if data is None:
        return {"error": "Stock data not found!"}

    scaler = MinMaxScaler(feature_range=(0,1))
    data['Scaled_Close'] = scaler.fit_transform(data[['Close']])

    x_input = data['Scaled_Close'].values[-100:].reshape(1, 100, 1)
    future_days = 365
    predictions = []
    dates = pd.date_range(start=pd.Timestamp.now(), periods=future_days).strftime('%Y-%m-%d').tolist()

    for _ in range(future_days):
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0, 0])
        x_input = np.append(x_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    prediction_data = [
        {
            "id": generate_unique_id(),
            "date": dates[i],
            "predicted_price": float(predictions[i][0])
        }
        for i in range(future_days)
    ]

    return {
        "stock_symbol": stock_symbol,
        "trend": "Up" if predictions[-1][0] > predictions[0][0] else "Down",
        "predictions": prediction_data
    }

# ✅ Yearly Analysis for Stock Graph with Unique ID Per Year
def get_yearly_analysis(stock_symbol):
    data = get_stock_data(stock_symbol, years=20)
    if data is None:
        return {"error": "Stock data not found!"}

    data['Year'] = data['Date'].dt.year

    yearly_summary = data.groupby('Year').agg({
        'Open': 'first',
        'Close': 'last',
        'High': 'max',
        'Low': 'min',
        'Volume': 'sum'
    }).reset_index()

    yearly_data = [
        {
            "id": generate_unique_id(),
            "year": int(row["Year"]),
            "open": float(row["Open"]),
            "close": float(row["Close"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "volume": int(row["Volume"])
        }
        for _, row in yearly_summary.iterrows()
    ]

    return {
        "stock_symbol": stock_symbol,
        "yearly_data": yearly_data
    }

# ✅ FastAPI Setup
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Stock Predictor API is running!"}

@app.get("/fetch/{stock_symbol}")
def fetch_stock(stock_symbol: str):
    data = get_stock_data(stock_symbol)
    if data is None:
        return {"error": "Stock data not found!"}

    stock_data = [
        {
            "id": generate_unique_id(),
            "date": str(row["Date"].date()),
            "open": float(row["Open"]),
            "close": float(row["Close"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "volume": int(row["Volume"])
        }
        for _, row in data.iterrows()
    ]

    return {
        "stock_symbol": stock_symbol,
        "data": stock_data
    }

@app.get("/predict/{stock_symbol}")
def predict_stock_route(stock_symbol: str):
    return predict_stock(stock_symbol)

@app.get("/yearly/{stock_symbol}")
def yearly_analysis(stock_symbol: str):
    return get_yearly_analysis(stock_symbol)

# ✅ Allow CORS for Frontend Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Entry point for Render
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))  # fallback for local dev
    uvicorn.run(app, host="0.0.0.0", port=port)
