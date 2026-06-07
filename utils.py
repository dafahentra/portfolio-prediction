from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import pandas as pd

def format_price(value):
    return f"${round(value, 2)}"

# RSI Calculation
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Sharpe Ratio
def sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

# Sortino Ratio
def sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    downside_returns = returns[returns < target_return]
    return (returns.mean() - target_return) / downside_returns.std()

# Portfolio Tracker
def track_portfolio(portfolio_tickers):
    portfolio = {ticker: yf.Ticker(ticker).history(period="10y")['Close'] for ticker in portfolio_tickers}
    portfolio_df = pd.DataFrame(portfolio)
    return portfolio_df



# Function to plot confusion matrix for Random Forest with custom color scheme
def plot_confusion_matrix_rf(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
    disp.plot(cmap="twilight")  # Custom color scheme
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

# Function to plot precision-recall curve for Random Forest with custom colors
def plot_precision_recall_curve_rf(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, marker='.', color='#FF5733', label=f'PR AUC = {pr_auc:.2f}')  # Custom color
    plt.title("Precision-Recall Curve - Random Forest")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot loss curve for LSTM with custom color scheme
def plot_loss_curve_lstm(history):
    plt.plot(history.history['loss'], label='Training Loss', color='#1f77b4')  # Custom color
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e')  # Custom color
    plt.title("LSTM Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def add_price_action_features(data):
    # 1. Base Technicals (RSI & MACD)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2

    # 2. Candlestick Patterns
    body = abs(data['Close'] - data['Open'])
    upper_shadow = data['High'] - data[['Open', 'Close']].max(axis=1)
    lower_shadow = data[['Open', 'Close']].min(axis=1) - data['Low']
    avg_body = body.rolling(window=14).mean()

    data['Is_Doji'] = (body < (0.1 * avg_body)).astype(int)
    data['Is_Hammer'] = ((body < (0.3 * (data['High'] - data['Low']))) &
                         (lower_shadow > (2 * body)) &
                         (upper_shadow < (0.1 * body))).astype(int)

    data['Bullish_Engulfing'] = ((data['Close'].shift(1) < data['Open'].shift(1)) & 
                                 (data['Open'] < data['Close']) & 
                                 (data['Open'] <= data['Close'].shift(1)) & 
                                 (data['Close'] >= data['Open'].shift(1))).astype(int)

    data['Bearish_Engulfing'] = ((data['Close'].shift(1) > data['Open'].shift(1)) & 
                                 (data['Open'] > data['Close']) & 
                                 (data['Open'] >= data['Close'].shift(1)) & 
                                 (data['Close'] <= data['Open'].shift(1))).astype(int)

    # 3. Chart Patterns: Moving Average Trends & Crossovers
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    data['Golden_Cross'] = ((data['SMA_50'] > data['SMA_200']) & (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))).astype(int)
    data['Death_Cross'] = ((data['SMA_50'] < data['SMA_200']) & (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))).astype(int)

    # 4. Chart Patterns: Bollinger Bands Breakouts
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    std_20 = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (std_20 * 2)
    data['BB_Lower'] = data['SMA_20'] - (std_20 * 2)
    
    data['BB_Breakout_Upper'] = (data['Close'] > data['BB_Upper']).astype(int)
    data['BB_Breakout_Lower'] = (data['Close'] < data['BB_Lower']).astype(int)

    return data

# Updated Random Forest training function with Price Action
def train_rf_model_with_graphs(data, ticker):
    # Add Price Action Features
    data = add_price_action_features(data)
    
    # Feature Engineering
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()

    # Splitting Data
    features = ['SMA_10', 'Return', 'Volume', 'RSI', 'MACD', 'Is_Doji', 'Is_Hammer', 
                'Bullish_Engulfing', 'Bearish_Engulfing', 'SMA_50', 'SMA_200', 
                'Golden_Cross', 'Death_Cross', 'BB_Breakout_Upper', 'BB_Breakout_Lower']
    
    # If Volume is not present (e.g. some indices), handle it gracefully
    if 'Volume' not in data.columns:
        data['Volume'] = 0

    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix_fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="pinkyl")  # Custom color scale

    # Generate Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    precision_recall_fig = go.Figure()
    precision_recall_fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall', line=dict(color='#FF5733')))  # Custom color
    precision_recall_fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")

    return model, accuracy, confusion_matrix_fig, precision_recall_fig

# Updated LSTM training function with Price Action
def train_lstm_model_with_graphs(data, ticker):
    # Add Price Action Features
    data = add_price_action_features(data)
    data = data.dropna()
    
    if 'Volume' not in data.columns:
        data['Volume'] = 0
        
    features = ['Close', 'Volume', 'RSI', 'MACD', 'Is_Doji', 'Is_Hammer', 
                'Bullish_Engulfing', 'Bearish_Engulfing', 'SMA_50', 'SMA_200', 
                'Golden_Cross', 'Death_Cross', 'BB_Breakout_Upper', 'BB_Breakout_Lower']
    
    # Train-Test Split Index
    train_size_idx = int(len(data) * 0.8)
    
    # Scale multivariate data (Fit scaler ONLY on training set to prevent Data Leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[features].iloc[:train_size_idx].values)
    
    # Transform entire dataset to create continuous sequences
    scaled_data = scaler.transform(data[features].values)

    # Create time series dataset for multivariate input
    def create_dataset(dataset, time_step=1):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), :]) # All features
            y.append(dataset[i + time_step, 0])     # Target is 'Close' (index 0)
        return np.array(X), np.array(y)

    time_step = 50  # Number of past data points used to predict the next value
    X, y = create_dataset(scaled_data, time_step)

    # Train-Test Split Arrays
    train_size_seq = int(len(X) * 0.8)
    X_train, X_test = X[:train_size_seq], X[train_size_seq:]
    y_train, y_test = y[:train_size_seq], y[train_size_seq:]

    # Build Multivariate LSTM Model
    model = Sequential()
    # X.shape[1] is time_step, X.shape[2] is number of features
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train Model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Generate Loss Curve
    loss_curve_fig = go.Figure()
    loss_curve_fig.add_trace(go.Scatter(x=list(range(1, len(history.history['loss']) + 1)),
                                        y=history.history['loss'],
                                        mode='lines',
                                        name='Training Loss',
                                        line=dict(color='#1f77b4')))  # Custom color for training loss
    loss_curve_fig.add_trace(go.Scatter(x=list(range(1, len(history.history['val_loss']) + 1)),
                                        y=history.history['val_loss'],
                                        mode='lines',
                                        name='Validation Loss',
                                        line=dict(color='#ff7f0e')))  # Custom color for validation loss
    loss_curve_fig.update_layout(title="Multivariate LSTM Training Loss Curve",
                                  xaxis_title="Epoch",
                                  yaxis_title="Loss")

    return model, history, loss_curve_fig, scaler, scaled_data[-50:]
