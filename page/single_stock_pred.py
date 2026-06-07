import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from utils import calculate_rsi, train_rf_model_with_graphs, train_lstm_model_with_graphs
import pandas as pd
import numpy as np


def _scale_feature(scaler, value, feature_idx):
    # Per-feature scaling using the scaler's stored min/max; clipped to [0, 1]
    if np.isnan(value):
        return 0.0
    feature_range = scaler.data_max_[feature_idx] - scaler.data_min_[feature_idx]
    if feature_range == 0:
        return 0.0
    return float(np.clip((value - scaler.data_min_[feature_idx]) / feature_range, 0.0, 1.0))


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
        with st.spinner("Training the model..."):
            model_lstm, _, loss_curve_fig, scaler, last_seq = train_lstm_model_with_graphs(data)
        model_detail.caption(":material/check_circle: Training complete")

        model_detail.subheader("LSTM Performance Graphs")
        with model_detail.popover("Training Loss Curve", use_container_width=True):
            st.plotly_chart(loss_curve_fig, use_container_width=True)

        # --- 7-Day Forward Predictions ---
        # Feature order in last_seq (from train_lstm_model_with_graphs):
        # [0] Close  [1] Volume  [2] RSI  [3] MACD
        # [4] Is_Doji  [5] Is_Hammer  [6] Bullish_Engulfing  [7] Bearish_Engulfing
        # [8] SMA_50  [9] SMA_200  [10] Golden_Cross  [11] Death_Cross
        # [12] BB_Breakout_Upper  [13] BB_Breakout_Lower
        #
        # RSI and MACD are recomputed each step from a rolling Close buffer.
        # Candlestick-based features (indices 4-7) need Open/High/Low which are
        # unavailable for future dates, so they are carried forward from the last day.

        pred.subheader("Upcoming Week Predictions")
        future_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1), periods=7
        )
        predictions = []
        behavior = []

        current_seq = last_seq.copy()
        last_real_close = data['Close'].iloc[-1]

        # Keep 250 real closes as a rolling buffer for RSI/MACD recomputation
        close_history = list(data['Close'].values[-250:])

        for _ in range(7):
            input_seq = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
            predicted_scaled_close = model_lstm.predict(input_seq, verbose=0)[0][0]

            # Inverse-transform Close only (MinMaxScaler is per-feature)
            pad_array = np.zeros((1, current_seq.shape[1]))
            pad_array[0, 0] = predicted_scaled_close
            predicted_price = float(scaler.inverse_transform(pad_array)[0][0])

            predictions.append(predicted_price)
            behavior.append("Increase" if predicted_price > last_real_close else "Decrease")
            last_real_close = predicted_price

            # Recompute RSI and MACD from the updated Close buffer
            close_history.append(predicted_price)
            cs = pd.Series(close_history)

            delta = cs.diff()
            gain = delta.where(delta > 0, 0.0)
            loss_s = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(com=13, min_periods=14).mean()
            avg_loss = loss_s.ewm(com=13, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            new_rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            new_macd = float(
                (cs.ewm(span=12, adjust=False).mean() - cs.ewm(span=26, adjust=False).mean()).iloc[-1]
            )

            new_day = current_seq[-1].copy()
            new_day[0] = predicted_scaled_close
            new_day[2] = _scale_feature(scaler, new_rsi, 2)
            new_day[3] = _scale_feature(scaler, new_macd, 3)
            current_seq = np.append(current_seq[1:], [new_day], axis=0)

        predictions_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": [f"${p:.2f}" for p in predictions],
            "Behavior": behavior
        })
        pred.dataframe(predictions_df, hide_index=True, use_container_width=True)