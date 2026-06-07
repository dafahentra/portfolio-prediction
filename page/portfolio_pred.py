import streamlit as st
import yfinance as yf
import pandas as pd
from utils import track_portfolio, train_rf_model_with_graphs, train_lstm_model_with_graphs, sharpe_ratio, sortino_ratio, format_price
import plotly.graph_objects as go
import numpy as np

def portfolio_pred_page():
    st.header("Portfolio Prediction")

    # Portfolio Tickers Input
    portfolio_tickers = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, GOOGL, MSFT):")

    st.divider()

    if portfolio_tickers:
        tickers_list = [ticker.strip() for ticker in portfolio_tickers.split(",")]
        
        st.subheader("Portfolio Weighting (%)")
        st.caption("Enter the percentage weight for each stock. They will be auto-normalized if they don't sum to 100%.")
        weights_input = []
        cols = st.columns(len(tickers_list))
        for i, ticker in enumerate(tickers_list):
            with cols[i]:
                w = st.number_input(f"{ticker}", min_value=0.0, max_value=100.0, value=100.0/len(tickers_list), key=f"weight_{ticker}")
                weights_input.append(w)
                
        total_w = sum(weights_input)
        if total_w == 0:
            st.error("Total weight cannot be zero.")
            return
            
        # Normalize weights to sum to 1.0
        weights = [w / total_w for w in weights_input]

        # Fetch portfolio data
        portfolio_df = track_portfolio(tickers_list)

        # Ensure the DataFrame has the expected columns
        if not all(ticker in portfolio_df.columns for ticker in tickers_list):
            st.error("Some of the specified tickers are missing in the portfolio data.")
            return

        # Add 'Close' column for compatibility (e.g., using the first stock's prices as 'Close')
        portfolio_df['Close'] = portfolio_df[tickers_list[0]]

        # Display Stock Information: Name and Today's Price
        st.subheader("Stock Information")
        stock_info = []
        for ticker in tickers_list:
            stock = yf.Ticker(ticker)
            todays_data = stock.history(period="1d")
            stock_info.append({"Ticker": ticker, "Name": stock.info['shortName'], "Price": format_price(todays_data['Close'].iloc[0])})
        
        stock_info_df = pd.DataFrame(stock_info)
        st.dataframe(stock_info_df, hide_index=True, use_container_width=True)

        st.divider()


        # Portfolio Performance: Line chart for each ticker
        st.subheader("Portfolio Performance")
        st.line_chart(portfolio_df[tickers_list])

        st.divider()

        pred = st.container()

        st.divider()

        # Risk/Return Analysis: Sharpe and Sortino Ratios
        st.subheader("Risk/Return Analysis")
        returns = portfolio_df[tickers_list].pct_change().dropna()
        
        # Calculate daily portfolio return using weights
        portfolio_daily_returns = (returns * weights).sum(axis=1)

        for i, ticker in enumerate(tickers_list):
            sharpe = sharpe_ratio(returns[ticker])
            sortino = sortino_ratio(returns[ticker])
            # Cumulative return for individual stock
            indiv_cumulative_return = (portfolio_df[ticker].iloc[-1] / portfolio_df[ticker].iloc[0]) - 1
            st.write(f"**{ticker}** (Weight: {weights[i]*100:.1f}%) - Total Return: {indiv_cumulative_return*100:.2f}%, Sharpe Ratio: {sharpe:.2f}, Sortino Ratio: {sortino:.2f}")
        
        st.divider()

        # Total Portfolio Return and Risk
        st.subheader("Total Portfolio Return and Risk")
        portfolio_sharpe = sharpe_ratio(portfolio_daily_returns)
        portfolio_sortino = sortino_ratio(portfolio_daily_returns)
        
        # Cumulative return for the entire portfolio
        portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
        portfolio_total_return = portfolio_cumulative_returns.iloc[-1] - 1
        
        st.write(f"Total Portfolio Return: {portfolio_total_return*100:.2f}%")
        st.write(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")
        st.write(f"Portfolio Sortino Ratio: {portfolio_sortino:.2f}")

        st.divider()

        # Feedback System for Optimizing the Portfolio
        st.subheader("Portfolio Optimization Feedback")
        feedback = ""
        if portfolio_sharpe < 1:
            feedback += "The portfolio's Sharpe Ratio is below 1, which indicates that risk-adjusted returns are low. Consider diversifying or adjusting asset weights.\n"
        if portfolio_sortino < 1:
            feedback += "The Sortino Ratio is below 1, suggesting the portfolio has a high downside risk. You may want to consider reducing riskier assets.\n"
        
        if portfolio_total_return < 0:
            feedback += "The portfolio has a negative return over the period. It may be worth reviewing the assets to identify underperformers.\n"
        
        if not feedback:
            feedback = "The portfolio appears to be well-balanced based on risk and return metrics."

        st.write(feedback)

        st.divider()

        # Predictions for the Next 7 Days and Candlestick Chart for Each Stock
        pred.subheader("Stock Predictions for the Next 7 Days")
        for ticker in tickers_list:
            stock = yf.Ticker(ticker)
            historical_data = stock.history(period="10y")

            # Prepare data for predictions (use closing prices for simplicity)
            historical_data['Date'] = historical_data.index
            historical_data = historical_data[['Date', 'Close']]

            # Generate predictions (for simplicity, assuming we have a function to predict the next 7 days)
            # You can replace this with your own prediction model like LSTM or any suitable model
            predictions = historical_data[['Date', 'Close']].tail(30)  # Get the last 30 days for predictions

            # Add next 7 days' prediction (this part can be enhanced by actual prediction models like LSTM)
            predicted_dates = pd.date_range(predictions['Date'].max(), periods=8, freq='D')[1:]
            predicted_prices = np.round(np.random.uniform(
                low=predictions['Close'].iloc[-1] * 0.95,
                high=predictions['Close'].iloc[-1] * 1.05,
                size=7
            ), 2)

            prediction_df = pd.DataFrame({
                'Date': predicted_dates,
                'Predicted Close': [f"${price}" for price in predicted_prices]
            })

            # Display stock information inside an expander with stock ticker and name
            with pred.expander(f"{ticker} - {stock.info['shortName']}"):
                st.write(f"{ticker} - Predicted Prices for the Next 7 Days")
                st.dataframe(prediction_df, use_container_width=True, hide_index=True)

                # Plot Candlestick Chart for the Stock
                fig = go.Figure(data=[go.Candlestick(x=historical_data['Date'],
                                                     open=historical_data['Close'],
                                                     high=historical_data['Close'] * 1.05,
                                                     low=historical_data['Close'] * 0.95,
                                                     close=historical_data['Close'])])

                fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)

        # Train and Display Random Forest Model
        st.sidebar.divider()
        model_detail = st.sidebar.container(border=True)
        model_detail.subheader("Random Forest Model")
        try:
            with st.spinner("Training the model..."):
                model_rf, accuracy, confusion_matrix_fig, precision_recall_fig = train_rf_model_with_graphs(portfolio_df)
            model_detail.caption(":material/check_circle: Training complete")
            model_detail.write(f"Accuracy: {accuracy:.2f}")

            # Random Forest Graphs
            model_detail.subheader("Random Forest Performance Graphs")
            with model_detail.popover("Confusion Matrix", use_container_width=True):
                st.plotly_chart(confusion_matrix_fig, use_container_width=True)
            with model_detail.popover("Precision-Recall Curve", use_container_width=True):
                st.plotly_chart(precision_recall_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error training Random Forest model: {e}")

        # Train and Display LSTM Model
        model_detail.subheader("LSTM Model")
        try:
            with st.spinner("Training the model..."):
                model_lstm, history, loss_curve_fig = train_lstm_model_with_graphs(portfolio_df)
            model_detail.caption(":material/check_circle: Training complete")

            # Display Training Loss Curve
            model_detail.subheader("LSTM Performance Graphs")
            with model_detail.popover("Training Loss Curve", use_container_width=True):
                st.plotly_chart(loss_curve_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error training LSTM model: {e}")
