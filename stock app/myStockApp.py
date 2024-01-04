import streamlit as st
import yfinance as yf
import pandas as pd

st.write("""
# Simple Stocke Price App
         
Shown are the stock closing price and volume of google!
"""
)

tickerSymbol = 'GOOGL'

tickerData = yf.Ticker(tickerSymbol)

tickerDf = tickerData.history(period='1d', start = '2010-5-31', end = '2020-5-31')

st.write("""
## Closing Price""")
st.line_chart(tickerDf.Close)
st.write("""
## Volume Price""")
st.line_chart(tickerDf.Volume)