import pandas as pd

def load_data():
    df = pd.read_csv("data/P587 DATASET.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def evaluate_models(df):
    # Your ARIMA, SARIMA, XGBoost, LSTM evaluation logic
    return pd.DataFrame({
        "Model": ["ARIMA", "SARIMA", "XGBoost", "LSTM"],
        "R² Score": [0.91, 0.93, 0.95, 0.96],
        "RMSE": [2.1, 1.9, 1.7, 1.6]
    })
