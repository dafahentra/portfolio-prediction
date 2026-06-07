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
    portfolio = {ticker: yf.Ticker(ticker).history(period="1y")['Close'] for ticker in portfolio_tickers}
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

# Updated Random Forest training function with custom colors for graphs
def train_rf_model_with_graphs(data):
    # Feature Engineering
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Return'] = data['Close'].pct_change()
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)
    data['Lag_3'] = data['Close'].shift(3)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()

    # Splitting Data
    X = data[['SMA_10', 'SMA_50', 'Return', 'Lag_1', 'Lag_2', 'Lag_3']]
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

# Updated LSTM training function with custom color for loss curve
def train_lstm_model_with_graphs(data):
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create time series dataset
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 50  # Number of past data points used to predict the next value
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train-Test Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
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
    loss_curve_fig.update_layout(title="LSTM Training Loss Curve",
                                  xaxis_title="Epoch",
                                  yaxis_title="Loss")

    return model, history, loss_curve_fig
