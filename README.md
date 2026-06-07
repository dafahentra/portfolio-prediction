# Quantitative Portfolio Prediction Dashboard

An institutional-grade portfolio analysis and stock prediction platform built with Python and Streamlit. This application combines modern financial metrics with a 13-dimensional Deep Learning architecture to forecast market movements based on pure Price Action mechanics.

Demo: https://portfolio-stock-pred.streamlit.app/

## Architecture

This project abandons traditional fundamental analysis in favor of high-density technical feature engineering, simulating the behavior of quantitative algorithmic trading systems. 

The machine learning pipeline (Random Forest & Multivariate LSTM) processes 13 dynamic market dimensions:
- **Momentum Indicators:** RSI, MACD, Volume
- **Short-Term Candlestick Patterns:** Doji, Hammer, Bullish/Bearish Engulfing
- **Long-Term Chart Patterns:** SMA 50, SMA 200, Golden Cross, Death Cross
- **Volatility Anomalies:** Bollinger Bands Breakout (Upper & Lower)

## Key Capabilities

- **Autoregressive Price Forecasting:** 7-day future price simulation using a Multivariate LSTM trained on strict, leak-free time-series scaling.
- **Risk-Return Profiling:** Automated calculation of Sharpe and Sortino ratios for individual assets and the aggregate portfolio.
- **Interactive Visualization:** Real-time Candlestick rendering and model performance evaluation graphs (Precision-Recall & Loss Curves) using Plotly.
- **Data Integrity:** Strict forward-only technical calculation to eliminate look-ahead bias and ensure realistic backtest environments.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-prediction.git
cd portfolio-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the dashboard:
```bash
streamlit run app.py
```

## Usage

Input a comma-separated list of stock tickers (e.g., AAPL, MSFT, TSLA) to initialize the dashboard. The system will automatically construct the 13-dimensional matrix, train the models locally, and output the actionable predictions and risk metrics.

## License

This project is licensed under the MIT License.
