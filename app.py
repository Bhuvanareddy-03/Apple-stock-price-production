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
# 📤 Upload CSV File
# ----------------------------------------------------------
st.title("🍏 Apple Stock Price Prediction")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

# ----------------------------------------------------------
# 📅 Prepare Dataset
# ----------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df.set_index('Date', inplace=True)
df = df.fillna(method='ffill')

# ----------------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------------
st.sidebar.title("📈 Forecast Settings")
model_choice = st.sidebar.selectbox("Choose Model for Forecast", ["ARIMA", "SARIMA", "XGBoost", "LSTM"])
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=10, max_value=60, value=30)

# ----------------------------------------------------------
# 📉 Historical Closing Prices
# ----------------------------------------------------------
st.subheader("📉 Historical Closing Prices")
st.line_chart(df['Close'])

# ----------------------------------------------------------
# Time Series Split
# ----------------------------------------------------------
train_size = int(len(df) * 0.9)
train, test = df['Close'][:train_size], df['Close'][train_size:]

# ----------------------------------------------------------
# ARIMA Model
# ----------------------------------------------------------
arima_model = ARIMA(train, order=(5,1,0))
arima_fit = arima_model.fit()
arima_pred = arima_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
arima_pred.index = test.index

# ----------------------------------------------------------
# SARIMA Model
# ----------------------------------------------------------
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit(disp=False)
sarima_pred = sarima_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
sarima_pred.index = test.index

# ----------------------------------------------------------
# XGBoost Model
# ----------------------------------------------------------
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

xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

# ----------------------------------------------------------
# LSTM Model
# ----------------------------------------------------------
data = df[['Close']]
scaler_lstm = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler_lstm.fit_transform(data)

train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train_lstm, y_train_lstm = create_dataset(train_data, time_step)
X_test_lstm, y_test_lstm = create_dataset(test_data, time_step)

X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

lstm_pred = model_lstm.predict(X_test_lstm)
lstm_pred = scaler_lstm.inverse_transform(lstm_pred)
y_test_actual = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1,1))

# ----------------------------------------------------------
# Model Comparison
# ----------------------------------------------------------
metrics = {
    'ARIMA': r2_score(test, arima_pred),
    'SARIMA': r2_score(test, sarima_pred),
    'XGBoost': r2_score(y_test, xgb_pred),
    'LSTM': r2_score(y_test_actual, lstm_pred)
}
rmse_vals = {
    'ARIMA': sqrt(mean_squared_error(test, arima_pred)),
    'SARIMA': sqrt(mean_squared_error(test, sarima_pred)),
    'XGBoost': sqrt(mean_squared_error(y_test, xgb_pred)),
    'LSTM': sqrt(mean_squared_error(y_test_actual, lstm_pred))
}

comparison_df = pd.DataFrame({
    'Model': list(metrics.keys()),
    'R² Score': list(metrics.values()),
    'RMSE': list(rmse_vals.values())
}).round(4)

st.subheader("📊 Model Performance Comparison (R² and RMSE)")
st.dataframe(comparison_df)

# ----------------------------------------------------------
# 🏆 Identify Best Model
# ----------------------------------------------------------
best_model = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
st.success(f"🏆 Best Performing Model: {best_model}")

# ----------------------------------------------------------
# 📈 Actual vs Predicted Visualization
# ----------------------------------------------------------
st.subheader("📉 Actual vs Predicted Closing Prices")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(test.index, test.values, label='Actual', color='black')
ax.plot(test.index, arima_pred, label='ARIMA', linestyle='--')
ax.plot(test.index, sarima_pred, label='SARIMA', linestyle='--')
ax.plot(y_test.index, xgb_pred, label='XGBoost', linestyle='--')
ax.plot(test.index[-len(lstm_pred):], lstm_pred, label='LSTM', linestyle='--')
ax.set_title("Actual vs Predicted Closing Prices")
ax.legend()
st.pyplot(fig)

# ----------------------------------------------------------
# Forecasting
# ----------------------------------------------------------
st.subheader(f"📅 {forecast_days}-Day Forecast using {model_choice}")
if model_choice == 'ARIMA':
    forecast = arima_fit.forecast(steps=forecast_days)
elif model_choice == 'SARIMA':
    forecast = sarima_fit.forecast(steps=forecast_days)
elif model_choice == 'XGBoost':
    forecast = xgb_model.predict(X_test_scaled[-forecast_days:])
elif model_choice == 'LSTM':
    last_60 = scaled_data[-60:]
    temp_input = list(last_60.reshape(1, -1)[0])
    lst_output = []
    for i in range(forecast_days):
        X_input = np.array(temp_input[-60:]).reshape(1, 60, 1)
        yhat = model_lstm.predict(X_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
    forecast = scaler_lstm.inverse_transform(np.array(lst_output).reshape(-1,1)).flatten()

future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': forecast}).set_index('Date')

st.line_chart(pd.concat([df['Close'], forecast_df['Predicted_Close']]))

st.write("🔮 Forecast Table")
st.dataframe(forecast_df.head(10))
