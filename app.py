import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
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

# ----------------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------------
st.sidebar.image("logo.png", width=150)
st.sidebar.title("📈 Forecasting Controls")
model_choice = st.sidebar.selectbox("Choose Model", ["ARIMA", "SARIMA", "XGBoost", "LSTM"])
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=10, max_value=60, value=30)

# ----------------------------------------------------------
# Load and Prepare Dataset
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("P587 DATASET.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
    df = df.fillna(method='ffill')
    return df

df = load_data()
st.title("🍏 Apple Stock Price Forecasting")
st.subheader("📉 Historical Closing Price")
st.line_chart(df['Close'])

# ----------------------------------------------------------
# Time Series Split
# ----------------------------------------------------------
train_size = int(len(df) * 0.9)
train, test = df['Close'][:train_size], df['Close'][train_size:]

# ----------------------------------------------------------
# Forecasting Logic
# ----------------------------------------------------------
st.subheader(f"📅 {forecast_days}-Day Forecast using {model_choice}")

if model_choice == 'ARIMA':
    model = ARIMA(train, order=(5,1,0)).fit()
    pred = model.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
    r2 = r2_score(test, pred)
    rmse = sqrt(mean_squared_error(test, pred))
    forecast = model.forecast(steps=forecast_days)

elif model_choice == 'SARIMA':
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    pred = model.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
    r2 = r2_score(test, pred)
    rmse = sqrt(mean_squared_error(test, pred))
    forecast = model.forecast(steps=forecast_days)

elif model_choice == 'XGBoost':
    df_ml = df.copy()
    df_ml['Day'] = df_ml.index.day
    df_ml['Month'] = df_ml.index.month
    df_ml['Year'] = df_ml.index.year
    df_ml['MA_5'] = df_ml['Close'].rolling(5).mean()
    df_ml['MA_10'] = df_ml['Close'].rolling(10).mean()
    df_ml = df_ml.dropna()

    X = df_ml[['Open','High','Low','Volume','Day','Month','Year','MA_5','MA_10']]
    y = df_ml['Close']
    split = int(len(X)*0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, random_state=42)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    rmse = sqrt(mean_squared_error(y_test, pred))
    forecast = model.predict(X_test_scaled[-forecast_days:])

elif model_choice == 'LSTM':
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

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))
    r2 = r2_score(y_test_actual, pred)
    rmse = sqrt(mean_squared_error(y_test_actual, pred))

    last_60 = scaled_data[-60:]
    temp_input = list(last_60.reshape(1, -1)[0])
    lst_output = []
    for i in range(forecast_days):
        X_input = np.array(temp_input[-60:]).reshape(1, 60, 1)
        yhat = model.predict(X_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
    forecast = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).flatten()

# ----------------------------------------------------------
# Display Forecast
# ----------------------------------------------------------
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': forecast}).set_index('Date')

st.line_chart(pd.concat([df['Close'], forecast_df['Predicted_Close']]))
st.write("🔮 Forecast Table")
st.dataframe(forecast_df.head(10))

st.markdown(f"**📌 R² Score:** {r2:.4f}  **📉 RMSE:** {rmse:.4f}")

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------
st.markdown("""
<hr>
<div style='text-align: center'>
    Made with ❤️ by Bhuvaneswari in Bangalore<br>
    <a href='https://github.com/yourusername/apple-stock-forecast' target='_blank'>GitHub Repo</a>
</div>
""", unsafe_allow_html=True)
