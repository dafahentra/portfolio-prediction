import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from utils import (
    calculate_rsi, train_rf_model_with_graphs,
    train_lstm_model_with_graphs, predict_next_7_days
)
import pandas as pd
import numpy as np


@st.cache_resource(show_spinner=False)
def _get_trained_lstm_single(ticker: str):
    """
    Train and cache the LSTM model for a given ticker.

    @st.cache_resource persists the Keras model object in memory across
    Streamlit re-runs without serialising it, so the model is trained at
    most once per ticker per server session.

    Raises ValueError if yfinance returns no data for the ticker,
    surfacing a readable error instead of a cryptic crash inside the
    training pipeline.
    """
    data = yf.Ticker(ticker).history(period="10y")
    if data.empty:
        raise ValueError(
            f"No historical data found for ticker '{ticker}'. "
            "Check that the symbol is valid."
        )
    return train_lstm_model_with_graphs(data)


def single_stock_page():
    st.header("Single Stock Prediction")

    ticker = st.text_input("Enter a stock ticker (e.g., AAPL):")

    st.divider()

    if ticker:
        stock = yf.Ticker(ticker)
        data = stock.history(period="10y")

        if data.empty:
            st.error(f"No data found for ticker '{ticker}'. Please enter a valid stock ticker.")
            return

        today_data = stock.history(period="1d")
        today_price = today_data['Close'].iloc[-1] if not today_data.empty else data['Close'].iloc[-1]
        stock_info = stock.info

        # --- Stock Information ---
        st.subheader("Stock Information")
        st.write(f"**Name**: {stock_info.get('longName', 'N/A')}")
        st.write(f"**Symbol**: {ticker}")
        st.write(f"**Today's Price**: ${today_price:.2f}")

        st.divider()

        pred = st.container()

        st.divider()

        # --- Candlestick Chart ---
        st.subheader(f"Stock Prices for {ticker}")
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        fig.update_layout(
            title=f'{ticker} Stock Prices', xaxis_title="Date", yaxis_title="Price"
        )
        st.plotly_chart(fig)

        st.divider()

        # --- RSI ---
        st.subheader("Relative Strength Index (RSI)")
        data['RSI'] = calculate_rsi(data)
        st.line_chart(data['RSI'])

        # --- Sidebar: Models ---
        st.sidebar.divider()
        model_detail = st.sidebar.container(border=True)

        model_detail.subheader("Random Forest Model")
        with st.spinner("Training the model..."):
            _, accuracy, confusion_matrix_fig, precision_recall_fig = train_rf_model_with_graphs(data)
        model_detail.caption(":material/check_circle: Training complete")
        model_detail.write(f"CV Accuracy (5-fold TimeSeriesCV): {accuracy:.2f}")

        model_detail.subheader("Random Forest Performance Graphs")
        with model_detail.popover("Confusion Matrix", use_container_width=True):
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)
        with model_detail.popover("Precision-Recall Curve", use_container_width=True):
            st.plotly_chart(precision_recall_fig, use_container_width=True)

        model_detail.subheader("LSTM Model")

        # Retrieve the LSTM from cache; training only occurs on the first call
        # for this ticker. Subsequent Streamlit re-runs use the cached model.
        try:
            with st.spinner(f"Preparing LSTM for {ticker}..."):
                model_lstm, _, loss_curve_fig, scaler, last_seq, lstm_metrics = \
                    _get_trained_lstm_single(ticker)
        except Exception as e:
            st.error(f"Could not prepare LSTM for {ticker}: {e}")
            return

        model_detail.caption(":material/check_circle: Training complete")

        # Test metrics are reported in USD (original price scale) for interpretability
        model_detail.write(f"Test RMSE: ${lstm_metrics['rmse']:.2f}")
        model_detail.write(f"Directional Accuracy: {lstm_metrics['directional_accuracy']:.1%}")

        model_detail.subheader("LSTM Performance Graphs")
        with model_detail.popover("Training Loss Curve", use_container_width=True):
            st.plotly_chart(loss_curve_fig, use_container_width=True)

        # --- 7-Day Forward Predictions ---
        pred.subheader("Upcoming Week Predictions")

        # pd.bdate_range produces business days only (Mon–Fri), matching the
        # trading-day cadence of the LSTM's autoregressive sequence.
        # pd.date_range would incorrectly label predictions on weekends.
        future_dates = pd.bdate_range(
            start=data.index[-1] + pd.Timedelta(days=1), periods=7
        )

        close_history = list(data['Close'].values[-250:])
        volume_mean = float(data['Volume'].mean()) if 'Volume' in data.columns else 0.0

        predictions, behavior = predict_next_7_days(
            model_lstm, scaler, last_seq,
            float(data['Close'].iloc[-1]),
            close_history,
            volume_mean
        )

        predictions_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": [f"${p:.2f}" for p in predictions],
            "Behavior": behavior
        })
        pred.dataframe(predictions_df, hide_index=True, use_container_width=True)