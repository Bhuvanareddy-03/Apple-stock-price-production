import streamlit as st
import pandas as pd
from utils import load_data, evaluate_models

st.title("📊 Model Performance Comparison")

df = load_data()
comparison_df = evaluate_models(df)
st.dataframe(comparison_df)
