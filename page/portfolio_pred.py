import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from utils import (
    track_portfolio, train_rf_model_with_graphs,
    train_lstm_model_with_graphs, sharpe_ratio, sortino_ratio, format_price
)
import plotly.graph_objects as go


@st.cache_resource(show_spinner=False)
def _get_trained_lstm_cached(ticker: str):
    # Cached per ticker so Streamlit doesn't retrain on every re-run
    data = yf.Ticker(ticker).history(period="10y")
    return train_lstm_model_with_graphs(data)


def _predict_next_7_days(model_lstm, scaler, last_seq, last_real_close):
    # Autoregressively predicts 7 days; each step feeds its own output as input
    predictions = []
    behavior = []
    current_seq = last_seq.copy()
    current_close = last_real_close

    for _ in range(7):
        input_seq = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        predicted_scaled_close = model_lstm.predict(input_seq, verbose=0)[0][0]

        # MinMaxScaler is feature-independent: zeros in other columns don't
        # affect the Close column's inverse transform
        pad_array = np.zeros((1, current_seq.shape[1]))
        pad_array[0, 0] = predicted_scaled_close
        predicted_price = scaler.inverse_transform(pad_array)[0][0]

        predictions.append(predicted_price)
        behavior.append("Increase" if predicted_price > current_close else "Decrease")
        current_close = predicted_price

        new_day = current_seq[-1].copy()
        new_day[0] = predicted_scaled_close
        current_seq = np.append(current_seq[1:], [new_day], axis=0)

    return predictions, behavior


def portfolio_pred_page():
    st.header("Portfolio Prediction")

    portfolio_tickers = st.text_input(
        "Enter stock tickers separated by commas (e.g., AAPL, GOOGL, MSFT):"
    )

    st.divider()

    if not portfolio_tickers:
        return

    tickers_list = [t.strip().upper() for t in portfolio_tickers.split(",")]

    # --- Portfolio Weighting ---
    st.subheader("Portfolio Weighting (%)")
    st.caption("Enter the percentage weight for each stock. They will be auto-normalized if they don't sum to 100%.")
    weights_input = []
    cols = st.columns(len(tickers_list))
    for i, ticker in enumerate(tickers_list):
        with cols[i]:
            w = st.number_input(
                f"{ticker}", min_value=0.0, max_value=100.0,
                value=100.0 / len(tickers_list), key=f"weight_{ticker}"
            )
            weights_input.append(w)

    total_w = sum(weights_input)
    if total_w == 0:
        st.error("Total weight cannot be zero.")
        return

    weights = [w / total_w for w in weights_input]

    # Close-only prices — used for the performance chart and risk metrics
    portfolio_df = track_portfolio(tickers_list)
    if not all(ticker in portfolio_df.columns for ticker in tickers_list):
        st.error("Some of the specified tickers are missing in the portfolio data.")
        return

    # --- Stock Information ---
    st.subheader("Stock Information")
    stock_info = []
    for ticker in tickers_list:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period="1d")
        stock_info.append({
            "Ticker": ticker,
            "Name": stock.info.get('shortName', ticker),
            "Price": format_price(todays_data['Close'].iloc[0])
        })
    st.dataframe(pd.DataFrame(stock_info), hide_index=True, use_container_width=True)

    st.divider()

    # --- Portfolio Performance ---
    st.subheader("Portfolio Performance")
    st.line_chart(portfolio_df[tickers_list])

    st.divider()

    # --- Risk / Return Analysis ---
    st.subheader("Risk/Return Analysis")
    returns = portfolio_df[tickers_list].pct_change().dropna()
    portfolio_daily_returns = (returns * weights).sum(axis=1)

    for i, ticker in enumerate(tickers_list):
        sharpe = sharpe_ratio(returns[ticker])
        sortino = sortino_ratio(returns[ticker])
        sortino_str = f"{sortino:.2f}" if not np.isnan(sortino) else "N/A"
        indiv_cumulative_return = (
            portfolio_df[ticker].iloc[-1] / portfolio_df[ticker].iloc[0]
        ) - 1
        st.write(
            f"**{ticker}** (Weight: {weights[i]*100:.1f}%) — "
            f"Total Return: {indiv_cumulative_return*100:.2f}%, "
            f"Sharpe Ratio: {sharpe:.2f}, "
            f"Sortino Ratio: {sortino_str}"
        )

    st.divider()

    # --- Total Portfolio Metrics ---
    st.subheader("Total Portfolio Return and Risk")
    portfolio_sharpe = sharpe_ratio(portfolio_daily_returns)
    portfolio_sortino = sortino_ratio(portfolio_daily_returns)

    portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    portfolio_total_return = portfolio_cumulative_returns.iloc[-1] - 1

    sortino_display = f"{portfolio_sortino:.2f}" if not np.isnan(portfolio_sortino) else "N/A"
    st.write(f"Total Portfolio Return: {portfolio_total_return*100:.2f}%")
    st.write(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")
    st.write(f"Portfolio Sortino Ratio: {sortino_display}")

    st.divider()

    # --- Portfolio Optimization Feedback ---
    st.subheader("Portfolio Optimization Feedback")
    feedback_lines = []
    if not np.isnan(portfolio_sharpe) and portfolio_sharpe < 1:
        feedback_lines.append(
            "The portfolio's Sharpe Ratio is below 1, indicating low risk-adjusted returns. "
            "Consider diversifying or adjusting asset weights."
        )
    if not np.isnan(portfolio_sortino) and portfolio_sortino < 1:
        feedback_lines.append(
            "The Sortino Ratio is below 1, suggesting high downside risk. "
            "Consider reducing exposure to more volatile assets."
        )
    if portfolio_total_return < 0:
        feedback_lines.append(
            "The portfolio has a negative return over the period. "
            "Review assets to identify underperformers."
        )
    if not feedback_lines:
        feedback_lines.append(
            "The portfolio appears well-balanced based on risk and return metrics."
        )
    for line in feedback_lines:
        st.write(line)

    st.divider()

    # --- 7-Day LSTM Predictions (one model per ticker, results cached) ---
    st.subheader("Stock Predictions for the Next 7 Days")

    # Full OHLCV per ticker — stored here to reuse in the sidebar RF section below,
    # avoiding a second identical API call for tickers_list[0]
    full_data: dict[str, pd.DataFrame] = {}

    for ticker in tickers_list:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(period="10y")
        full_data[ticker] = historical_data

        with st.spinner(f"Preparing LSTM for {ticker}..."):
            model_lstm, _, _, scaler, last_seq = _get_trained_lstm_cached(ticker)

        last_real_close = historical_data['Close'].iloc[-1]
        preds, behavior = _predict_next_7_days(model_lstm, scaler, last_seq, last_real_close)

        future_dates = pd.date_range(
            start=historical_data.index[-1] + pd.Timedelta(days=1), periods=7
        )
        prediction_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Predicted Close': [f"${p:.2f}" for p in preds],
            'Behavior': behavior
        })

        with st.expander(f"{ticker} — {stock.info.get('shortName', ticker)}"):
            st.write(f"{ticker} — Predicted Prices for the Next 7 Days")
            st.dataframe(prediction_df, use_container_width=True, hide_index=True)

            # Candlestick chart using actual OHLC data
            fig = go.Figure(data=[go.Candlestick(
                x=historical_data.index,
                open=historical_data['Open'],
                high=historical_data['High'],
                low=historical_data['Low'],
                close=historical_data['Close']
            )])
            fig.update_layout(
                title=f"{ticker} Candlestick Chart",
                xaxis_title="Date", yaxis_title="Price"
            )
            st.plotly_chart(fig)

    # --- Sidebar: Model Details ---
    st.sidebar.divider()
    model_detail = st.sidebar.container(border=True)

    # Reuse the data already fetched in the prediction loop above
    first_ticker_full_data = full_data[tickers_list[0]]

    model_detail.subheader("Random Forest Model")
    try:
        with st.spinner("Training Random Forest..."):
            _, accuracy, confusion_matrix_fig, precision_recall_fig = train_rf_model_with_graphs(
                first_ticker_full_data
            )
        model_detail.caption(":material/check_circle: Training complete")
        model_detail.write(f"CV Accuracy (5-fold TimeSeriesCV): {accuracy:.2f}")

        model_detail.subheader("Random Forest Performance Graphs")
        with model_detail.popover("Confusion Matrix", use_container_width=True):
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)
        with model_detail.popover("Precision-Recall Curve", use_container_width=True):
            st.plotly_chart(precision_recall_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error training Random Forest model: {e}")

    model_detail.subheader("LSTM Model")
    try:
        # Reuse cached model — no retraining
        _, _, loss_curve_fig, _, _ = _get_trained_lstm_cached(tickers_list[0])
        model_detail.caption(":material/check_circle: Training complete")

        model_detail.subheader("LSTM Performance Graphs")
        with model_detail.popover("Training Loss Curve", use_container_width=True):
            st.plotly_chart(loss_curve_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")