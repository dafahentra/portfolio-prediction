# 📈 Portfolio Analyzer and Stock Prediction Dashboard 📉

## Overview

This Streamlit application provides a comprehensive platform for tracking portfolio performance, conducting risk-return analysis, generating stock price predictions, and visualizing financial data with interactive charts. The app leverages modern machine learning techniques and financial metrics to assist users in making informed decisions.

https://portfolio-stock-pred.streamlit.app/

[![Watch the video](./image.png)](https://youtu.be/BNgE11T8JC0)

## Features

- **Portfolio Tracking**: Input multiple stock tickers to track portfolio performance over time.
- **Risk-Return Analysis**: Calculate Sharpe and Sortino ratios for individual stocks and the entire portfolio.
- **Stock Information**: View real-time information such as stock name and current price for each ticker.
- **Stock Price Predictions**: Predict future stock prices using machine learning models (Random Forest and LSTM).
- **Candlestick Charts**: Visualize stock price movements with interactive candlestick charts.
- **Portfolio Optimization Feedback**: Get actionable feedback based on portfolio risk and return metrics.
- **Model Performance Insights**: Evaluate the accuracy of prediction models with performance graphs.

## Installation

1. Clone the repository:
   ```
   git clone <repository_url>
   ```

2. Navigate to the project directory:
   ```
   cd portfolio-prediction
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open the app in your browser and interact with the various sections:
   - **Portfolio Prediction**: Input stock tickers separated by commas.
   - **Risk/Return Analysis**: View Sharpe and Sortino ratios for each stock and the portfolio.
   - **Predictions and Charts**: Explore 7-day stock price predictions and candlestick charts.

## File Structure

- **app.py**: Main application file.
- **utils.py**: Contains utility functions for portfolio tracking, financial calculations, and machine learning models.
- **requirements.txt**: Lists all dependencies required for the app.

## Key Components

### Portfolio Tracking
- Input stock tickers and view their performance in a line chart.
- Real-time stock information is displayed in expanders labeled with the stock ticker and name.

### Risk-Return Analysis
- Compute and display Sharpe and Sortino ratios for each stock and the portfolio.
- Provide actionable feedback for optimizing the portfolio.

### Stock Price Predictions
- Train and visualize the performance of Random Forest and LSTM models.
- View predictions for the next 7 days with accompanying interactive candlestick charts.

### Visualization
- Generate interactive plots using Plotly for better insights into financial data.
- Candlestick charts offer a clear view of stock price movements.


## License

This project is licensed under the [MIT License](LICENSE).

