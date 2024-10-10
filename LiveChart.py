import streamlit as st
import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import time

# Set up Alpaca credentials
api_key = "PKAWTMUR165K9JGLAL71"
api_secret = "z8LFSRtbad5hrBX8vKIkqp1qAOTKIxfEWBWKJbDQ"
base_url = "https://paper-api.alpaca.markets"  # Paper trading URL (replace with live URL if needed)

# Initialize the Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

# Set page layout to wide (full screen width)
st.set_page_config(layout="wide")

# Add some CSS to remove padding/margin and allow full-width charts
st.markdown("""
    <style>
        .main {
            padding: 0px;
        }
        iframe {
            border: none;
        }
        .block-container {
            padding: 0rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app setup
st.title("Unlimited Backtesting")

# User input to choose stock symbol
ticker = st.text_input('Enter Stock Ticker', 'XAUUSD')

# If the user provides a ticker symbol
if ticker:
    # Use Streamlit's columns to split the canvas into two parts
    col1, col2 = st.columns(2)

    # Adjust the chart sizes within columns to avoid cuts
    chart_height = 600
    chart_width = "100%"

    # Display the first chart with 15-minute timeframe in the first column
    with col1:
        st.markdown(f"""
        <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_eb9b7&symbol={ticker}&interval=15&theme=light&style=1&toolbarbg=f1f3f6&studies=[]&withdateranges=true&hide_side_toolbar=false&allow_symbol_change=true&save_image=false&show_popup_button=false&locale=en" width="{chart_width}" height="{chart_height}" frameborder="0" allowfullscreen></iframe>
        """, unsafe_allow_html=True)

    # Display the second chart with 1-minute timeframe in the second column
    with col2:
        st.markdown(f"""
        <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_eb9b7&symbol={ticker}&interval=1&theme=light&style=1&toolbarbg=f1f3f6&studies=[]&withdateranges=true&hide_side_toolbar=false&allow_symbol_change=true&save_image=false&show_popup_button=false&locale=en" width="{chart_width}" height="{chart_height}" frameborder="0" allowfullscreen></iframe>
        """, unsafe_allow_html=True)

    # Function to fetch real-time data from Alpaca
    # Function to fetch real-time data from Alpaca
    # Function to fetch real-time data from Alpaca
    def get_stock_data(symbol, timeframe='1Min', limit=100):
        barset = api.get_bars(symbol, timeframe, limit=limit)

        # Convert barset to a list of bars
        bars = list(barset)

        # Check if bars are returned and process them
        if len(bars) > 0:
            return bars
        else:
            return []


    # Smart Money Concept (SMC) Indicator
    def smc_indicator(data):
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame({
            'Time': [bar.t for bar in data],
            'Open': [bar.o for bar in data],
            'High': [bar.h for bar in data],
            'Low': [bar.l for bar in data],
            'Close': [bar.c for bar in data],
            'Volume': [bar.v for bar in data]
        })

        # Calculate liquidity zones and Break of Structure (BOS)
        liquidity_high = df['High'].rolling(window=20).max()  # 20-period highest high
        liquidity_low = df['Low'].rolling(window=20).min()  # 20-period lowest low
        df['BOS'] = np.where((df['Close'] > liquidity_high.shift(1)), 'Break Up',
                             np.where(df['Close'] < liquidity_low.shift(1), 'Break Down', 'Neutral'))

        df['Liquidity High'] = liquidity_high
        df['Liquidity Low'] = liquidity_low

        return df

    # Create a placeholder for real-time data updates
    chart_placeholder = st.empty()

    # Real-time data fetching and indicator integration
    while True:
        # Fetch real-time data from Alpaca
        stock_data = get_stock_data(ticker, timeframe='1Min', limit=100)

        # Calculate the SMC indicator
        stock_data_with_smc = smc_indicator(stock_data)

        # Display real-time stock data and indicators
        st.subheader(f"Live Data for {ticker} (1Min)")
        st.write(
            stock_data_with_smc[['Time', 'Open', 'High', 'Low', 'Close', 'Liquidity High', 'Liquidity Low', 'BOS']])

        # Update the placeholder with the new data
        chart_placeholder.table(stock_data_with_smc[['Time', 'Open', 'Close', 'BOS']].tail(10))

        # Wait 60 seconds before fetching new data
        time.sleep(60)
