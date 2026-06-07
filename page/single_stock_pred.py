import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from utils import calculate_rsi, train_rf_model_with_graphs, train_lstm_model_with_graphs
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def single_stock_page():
    st.header("Single Stock Prediction")

    # Stock Ticker Input
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL):")

    st.divider()

    if ticker:
        # Fetch stock data from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period="10y")
        
        if data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please enter a valid stock ticker.")
            return
            
        historical_data = stock.history(period="10y")
        today_data = stock.history(period="1d")
        if not today_data.empty:
            today_price = today_data['Close'][-1]
        else:
            today_price = data['Close'][-1] # Fallback to last known price
            
        stock_info = stock.info

        # Display stock information
        st.subheader("Stock Information")
        st.write(f"**Name**: {stock_info.get('longName', 'N/A')}")
        st.write(f"**Symbol**: {ticker}")
        st.write(f"**Today's Price**: ${today_price:.2f}")

        st.divider()

        pred = st.container()

        st.divider()

        # Plotting Stock Prices
        st.subheader(f"Stock Prices for {ticker}")
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        st.divider()

        # Add RSI Calculation
        st.subheader("Relative Strength Index (RSI)")
        data['RSI'] = calculate_rsi(data)
        st.line_chart(data['RSI'])

        # Train and Display Random Forest Model
        st.sidebar.divider()

        model_detail = st.sidebar.container(border=True)
        model_detail.subheader("Random Forest Model")
        with st.spinner("Training the model..."):
            model_rf, accuracy, confusion_matrix_fig, precision_recall_fig = train_rf_model_with_graphs(data, ticker)
        model_detail.caption(":material/check_circle: Training complete")
        model_detail.write(f"Accuracy: {accuracy:.2f}")

        # Random Forest Graphs (Confusion Matrix and Precision-Recall Curve)
        model_detail.subheader("Random Forest Performance Graphs")
        with model_detail.popover("Confusion Matrix", use_container_width=True):
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)
        with model_detail.popover("Precision-Recall Curve", use_container_width=True):
            st.plotly_chart(precision_recall_fig, use_container_width=True)

        # Train and Display LSTM Model
        model_detail.subheader("LSTM Model")
        with st.spinner("Training the model..."):
            model_lstm, history, loss_curve_fig, scaler, last_seq = train_lstm_model_with_graphs(data, ticker)
        model_detail.caption(":material/check_circle: Training complete")

        # Display Training Loss Curve
        model_detail.subheader("LSTM Performance Graphs")
        with model_detail.popover("Training Loss Curve", use_container_width=True):
            st.plotly_chart(loss_curve_fig, use_container_width=True)

        # Predict next week's prices and behavior
        pred.subheader("Upcoming Week Predictions")
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
        predictions = []
        behavior = []

        # Use LSTM for prediction
        current_seq = last_seq.copy()
        last_real_close = data['Close'].iloc[-1]

        for i in range(7):
            input_seq = current_seq.reshape(1, 50, current_seq.shape[1])  # 3D shape
            predicted_scaled_close = model_lstm.predict(input_seq)[0][0]
            
            # To inverse transform, we need an array of the same feature dimension. We pad with zeros.
            pad_array = np.zeros((1, current_seq.shape[1]))
            pad_array[0, 0] = predicted_scaled_close
            predicted_price = scaler.inverse_transform(pad_array)[0][0]
            
            predictions.append(predicted_price)
            behavior.append("Increase" if predicted_price > last_real_close else "Decrease")
            last_real_close = predicted_price
            
            # Update the sequence: remove oldest day, append new day (keeping other features constant)
            new_day = current_seq[-1].copy()
            new_day[0] = predicted_scaled_close
            current_seq = np.append(current_seq[1:], [new_day], axis=0)

        # Create DataFrame for predictions
        predictions_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": [f"${pred:.2f}" for pred in predictions],
            "Behavior": behavior
        })
        pred.dataframe(predictions_df, hide_index=True, use_container_width=True)

