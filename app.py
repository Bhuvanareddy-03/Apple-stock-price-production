import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# Page Config
# ----------------------------------------------------------
st.set_page_config(page_title="Apple Stock Forecast", page_icon="🍏", layout="wide")
st.title("🍏 Apple Stock Price Forecasting with LSTM")

# ----------------------------------------------------------
# Upload Dataset
# ----------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload your stock CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
    df = df.fillna(method='ffill')

    st.subheader("📉 Historical Closing Price")
    st.line_chart(df['Close'])

    # ----------------------------------------------------------
    # Train-Test Split
    # ----------------------------------------------------------
    train_size = int(len(df) * 0.9)
    data = df[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_dataset(dataset, time_step=60):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            X.append(dataset[i:(i+time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 60
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - time_step:]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # ----------------------------------------------------------
    # Train LSTM Model
    # ----------------------------------------------------------
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

    # ----------------------------------------------------------
    # Predict and Evaluate
    # ----------------------------------------------------------
    pred = model.predict(X_test)
    pred_actual = scaler.inverse_transform(pred)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

    r2 = r2_score(y_test_actual, pred_actual)
    rmse = sqrt(mean_squared_error(y_test_actual, pred_actual))

    # ----------------------------------------------------------
    # Visualize Actual vs Predicted
    # ----------------------------------------------------------
    st.subheader("📊 Actual vs Predicted Closing Prices")
    result_df = pd.DataFrame({
        'Actual': y_test_actual.flatten(),
        'Predicted': pred_actual.flatten()
    }, index=df.index[-len(y_test_actual):])

    st.line_chart(result_df)

    st.markdown(f"**📌 R² Score:** {r2:.4f}  **📉 RMSE:** {rmse:.4f}")
else:
    st.info("Please upload a CSV file with stock data to begin.")
