import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from utils import (
    train_rf_model_with_graphs,
    train_lstm_model_with_graphs, sharpe_ratio, sortino_ratio, format_price,
    predict_next_7_days
)
import plotly.graph_objects as go


@st.cache_data(show_spinner=False, ttl=3600)
def _load_portfolio_history(tickers: tuple) -> dict[str, pd.DataFrame]:
    """
    Download full OHLCV data for all tickers in a single batched API call.
    Returns a per-ticker dict of DataFrames keyed by symbol.

    Using one yf.download call instead of N separate yf.Ticker().history()
    calls reduces API round-trips and avoids re-downloading on every Streamlit
    re-run thanks to @st.cache_data (TTL: 1 hour).

    For a single-ticker download, yfinance may return flat or MultiIndex columns
    depending on the version; the isinstance check normalises both cases.
    For multi-ticker downloads, yf.download returns a (metric, ticker) MultiIndex
    which xs() slices into per-ticker DataFrames with metric columns.
    """
    raw = yf.download(list(tickers), period="10y", progress=False)
    result = {}
    for ticker in tickers:
        if len(tickers) == 1:
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
        else:
            df = raw.xs(ticker, axis=1, level=1).copy()
        result[ticker] = df.dropna(how="all")
    return result


@st.cache_resource(show_spinner=False)
def _get_trained_lstm_cached(ticker: str):
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

    # Download full OHLCV for all tickers in one batched call; result is cached
    # for 1 hour to avoid repeated API calls across Streamlit re-runs.
    # try/except catches KeyError (invalid ticker not found in download result)
    # before it propagates past the graceful error check below.
    try:
        full_data = _load_portfolio_history(tuple(tickers_list))
    except Exception as e:
        st.error(f"Failed to load market data: {e}. Please verify all ticker symbols.")
        return

    if not all(t in full_data and not full_data[t].empty for t in tickers_list):
        st.error("Some tickers returned no data. Please verify all symbols are valid.")
        return

    # Build a Close-only DataFrame for portfolio-level analytics
    portfolio_df = pd.DataFrame({t: full_data[t]['Close'] for t in tickers_list})

    # --- Stock Information ---
    st.subheader("Stock Information")
    stock_info = []
    for ticker in tickers_list:
        stock = yf.Ticker(ticker)
        # Last known Close comes from the already-downloaded full_data;
        # only stock.info (metadata) requires a separate yfinance call
        last_close = full_data[ticker]['Close'].iloc[-1]
        stock_info.append({
            "Ticker": ticker,
            "Name": stock.info.get('shortName', ticker),
            "Price": format_price(last_close)
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

    for ticker in tickers_list:
        # Use OHLCV data from the already-cached batched download
        historical_data = full_data[ticker]

        try:
            with st.spinner(f"Preparing LSTM for {ticker}..."):
                model_lstm, _, _, scaler, last_seq, _ = _get_trained_lstm_cached(ticker)
        except Exception as e:
            st.error(f"Could not prepare LSTM for {ticker}: {e}")
            continue

        close_history = list(historical_data['Close'].values[-250:])
        volume_mean = (
            float(historical_data['Volume'].mean())
            if 'Volume' in historical_data.columns else 0.0
        )

        # Derive last_real_close from last_seq via inverse_transform instead of
        # historical_data['Close'].iloc[-1]. Both should be identical in practice,
        # but last_seq comes from the LSTM training download (yf.Ticker().history)
        # while historical_data comes from _load_portfolio_history (yf.download) —
        # two different API paths that could diverge by one trading day due to
        # cache age or timezone handling. Using inverse_transform guarantees the
        # comparison price is exactly what the LSTM saw last, keeping
        # Increase / Decrease labels internally consistent with the model's scale.
        # With clip=True: if the actual last Close exceeded the training maximum,
        # this gives the training-period peak (the LSTM's effective upper bound).
        pad_last = np.zeros((1, last_seq.shape[1]))
        pad_last[0, 0] = float(last_seq[-1, 0])
        last_real_close = float(scaler.inverse_transform(pad_last)[0][0])
        preds, behavior = predict_next_7_days(
            model_lstm, scaler, last_seq, last_real_close, close_history, volume_mean
        )

        # pd.bdate_range produces business days only (Mon–Fri), matching the
        # trading-day cadence of the LSTM's autoregressive sequence.
        # pd.date_range would incorrectly label predictions on weekends.
        future_dates = pd.bdate_range(
            start=historical_data.index[-1] + pd.Timedelta(days=1), periods=7
        )
        prediction_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Predicted Close': [f"${p:.2f}" for p in preds],
            'Behavior': behavior
        })

        stock = yf.Ticker(ticker)
        with st.expander(f"{ticker} — {stock.info.get('shortName', ticker)}"):
            st.write(f"{ticker} — Predicted Prices for the Next 7 Days")
            st.dataframe(prediction_df, use_container_width=True, hide_index=True)

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

    # RF and LSTM metrics are shown for the first ticker in the portfolio
    first_ticker_data = full_data.get(tickers_list[0])

    model_detail.subheader("Random Forest Model")
    if first_ticker_data is not None and not first_ticker_data.empty:
        try:
            with st.spinner("Training Random Forest..."):
                _, accuracy, confusion_matrix_fig, precision_recall_fig = train_rf_model_with_graphs(
                    first_ticker_data
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
        # Retrieve cached LSTM for the first ticker to display training metrics
        _, _, loss_curve_fig, _, _, lstm_metrics = _get_trained_lstm_cached(tickers_list[0])
        model_detail.caption(":material/check_circle: Training complete")

        # Test metrics are reported in USD (original price scale) for interpretability
        model_detail.write(f"Test RMSE: ${lstm_metrics['rmse']:.2f}")
        model_detail.write(f"Directional Accuracy: {lstm_metrics['directional_accuracy']:.1%}")

        model_detail.subheader("LSTM Performance Graphs")
        with model_detail.popover("Training Loss Curve", use_container_width=True):
            st.plotly_chart(loss_curve_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")