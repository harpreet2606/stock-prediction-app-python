import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def predict_stock_future(stock_data):
    try:
        # Feature engineering
        stock_data['Date'] = stock_data.index
        stock_data['Date'] = stock_data['Date'].astype(np.int64) // 10**9  # Convert to Unix timestamp
        X = stock_data[['Date']].values
        y = stock_data['Close'].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return rmse, stock_data
    except Exception as e:
        st.error(f"Error predicting future: {e}")
        return None, None

def analyze_profitability(stock_data):
    try:
        # Calculate percentage change over different time intervals
        stock_data['1_day_change'] = stock_data['Close'].pct_change(periods=1)
        stock_data['7_day_change'] = stock_data['Close'].pct_change(periods=7)
        stock_data['30_day_change'] = stock_data['Close'].pct_change(periods=30)

        # Calculate moving averages
        stock_data['50_day_MA'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['200_day_MA'] = stock_data['Close'].rolling(window=200).mean()

        # Determine trend based on moving averages
        stock_data['Trend'] = np.where(stock_data['50_day_MA'] > stock_data['200_day_MA'], 'Bullish', 'Bearish')

        # Determine profitability
        one_day_profitable = stock_data['1_day_change'].iloc[-1] > 0
        seven_day_profitable = stock_data['7_day_change'].iloc[-1] > 0
        thirty_day_profitable = stock_data['30_day_change'].iloc[-1] > 0

        return one_day_profitable, seven_day_profitable, thirty_day_profitable, stock_data.iloc[-1]['Close'], stock_data.iloc[-1]['Trend']
    except Exception as e:
        st.error(f"Error analyzing profitability: {e}")
        return None, None, None, None, None
 
def main():
    st.title("Stock Analysis App")
    
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple):")
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=3)
    
    if st.button("Get Analysis"):
        stock_data = get_stock_data(symbol, start_date, end_date)
        if stock_data is not None:
            rmse, stock_data = predict_stock_future(stock_data)
            if rmse is not None:
                st.subheader("Root Mean Squared Error (RMSE):")
                st.write(rmse)
                if stock_data is not None:
                    st.subheader("Stock Data:")
                    st.write(stock_data.tail())

                    st.subheader("Profitability Analysis:")
                    one_day_profitable, seven_day_profitable, thirty_day_profitable, current_price, trend = analyze_profitability(stock_data)
                    if one_day_profitable is not None and seven_day_profitable is not None and thirty_day_profitable is not None:
                        st.write("Is the stock potentially profitable:")
                        st.write(f"- For 1 day: {'Yes' if one_day_profitable else 'No'}")
                        st.write(f"- For 7 days: {'Yes' if seven_day_profitable else 'No'}")
                        st.write(f"- For 30 days: {'Yes' if thirty_day_profitable else 'No'}")

                    if trend is not None:
                        st.write(f"Current Price: ${current_price:.2f}")
                        st.write(f"Trend: {trend}")

if __name__ == "__main__":
    main()
