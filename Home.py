import streamlit as st

st.set_page_config(page_title="Stock Forecast Hub", page_icon="📈", layout="wide")

st.image("logo.png", width=120)
st.title("📊 Welcome to Stock Forecast Hub")
st.markdown("""
This app provides **multi-model stock price forecasting** using ARIMA, SARIMA, XGBoost, and LSTM.

Navigate through the pages on the left to:
- 📈 Compare model performance
- 🔮 Generate forecasts
- 📚 Learn about the models
""")
