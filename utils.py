from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import pandas as pd


def format_price(value):
    return f"${round(value, 2)}"


def calculate_rsi(data, window=14):
    # Wilder's EWM (com = window-1 ≡ alpha = 1/window); eps prevents division by zero
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))


def sharpe_ratio(returns, risk_free_rate=0.02):
    # Convert annual rate to daily, then annualize the result with sqrt(252)
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    std = excess_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess_returns.mean() / std) * np.sqrt(252)


def sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    # Only penalizes downside volatility; annualized with sqrt(252)
    daily_rf = risk_free_rate / 252
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return np.nan
    downside_std = downside_returns.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return ((returns.mean() - daily_rf) / downside_std) * np.sqrt(252)


def track_portfolio(portfolio_tickers):
    # Returns Close prices per ticker; used for performance chart and risk metrics
    portfolio = {
        ticker: yf.Ticker(ticker).history(period="10y")['Close']
        for ticker in portfolio_tickers
    }
    return pd.DataFrame(portfolio)


def add_price_action_features(data):
    # Work on a copy to avoid mutating the caller's DataFrame
    data = data.copy()

    # RSI via Wilder's EWM and MACD (12/26 EMA difference)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    data['RSI'] = 100 - (100 / (1 + rs))

    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2

    # Candlestick pattern flags
    body = abs(data['Close'] - data['Open'])
    upper_shadow = data['High'] - data[['Open', 'Close']].max(axis=1)
    lower_shadow = data[['Open', 'Close']].min(axis=1) - data['Low']
    avg_body = body.rolling(window=14).mean()

    data['Is_Doji'] = (body < (0.1 * avg_body)).astype(int)

    # Guard (body > avg_body * 0.1) prevents Doji from being flagged as Hammer;
    # when body ≈ 0, upper_shadow < 0.3 * body was trivially True
    data['Is_Hammer'] = (
        (body > (avg_body * 0.1)) &
        (body < (0.3 * (data['High'] - data['Low']))) &
        (lower_shadow > (2 * body)) &
        (upper_shadow < (0.3 * body))
    ).astype(int)

    data['Bullish_Engulfing'] = (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Open'] < data['Close']) &
        (data['Open'] <= data['Close'].shift(1)) &
        (data['Close'] >= data['Open'].shift(1))
    ).astype(int)

    data['Bearish_Engulfing'] = (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Open'] >= data['Close'].shift(1)) &
        (data['Close'] <= data['Open'].shift(1))
    ).astype(int)

    # Moving average trends and Golden/Death Cross signals
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Golden_Cross'] = (
        (data['SMA_50'] > data['SMA_200']) &
        (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))
    ).astype(int)
    data['Death_Cross'] = (
        (data['SMA_50'] < data['SMA_200']) &
        (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))
    ).astype(int)

    # Bollinger Bands (20-day, ±2σ) breakout flags
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    std_20 = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (std_20 * 2)
    data['BB_Lower'] = data['SMA_20'] - (std_20 * 2)
    data['BB_Breakout_Upper'] = (data['Close'] > data['BB_Upper']).astype(int)
    data['BB_Breakout_Lower'] = (data['Close'] < data['BB_Lower']).astype(int)

    return data


def train_rf_model_with_graphs(data):
    # Binary target: 1 if next-day Close > today's Close, else 0
    data = add_price_action_features(data)
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()

    features = [
        'SMA_10', 'Return', 'Volume', 'RSI', 'MACD',
        'Is_Doji', 'Is_Hammer', 'Bullish_Engulfing', 'Bearish_Engulfing',
        'SMA_50', 'SMA_200', 'Golden_Cross', 'Death_Cross',
        'BB_Breakout_Upper', 'BB_Breakout_Lower'
    ]
    if 'Volume' not in data.columns:
        data['Volume'] = 0

    X = data[features]
    y = data['Target']

    # 5-fold TimeSeriesSplit CV for a reliable accuracy estimate;
    # reported accuracy is the mean across folds
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42
        )
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        cv_scores.append(accuracy_score(y.iloc[val_idx], clf.predict(X.iloc[val_idx])))

    # Final model on 80/20 temporal split — used only for the graph visuals
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = float(np.mean(cv_scores))

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix_fig = px.imshow(
        cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="pinkyl"
    )

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    precision_recall_fig = go.Figure()
    precision_recall_fig.add_trace(go.Scatter(
        x=recall, y=precision, mode='lines',
        name='Precision-Recall', line=dict(color='#FF5733')
    ))
    precision_recall_fig.update_layout(
        title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision"
    )

    return model, accuracy, confusion_matrix_fig, precision_recall_fig


def train_lstm_model_with_graphs(data):
    # Multivariate LSTM: 14 features, 50-day lookback, predicts next Close
    data = add_price_action_features(data)
    data = data.dropna()

    if 'Volume' not in data.columns:
        data['Volume'] = 0

    features = [
        'Close', 'Volume', 'RSI', 'MACD',
        'Is_Doji', 'Is_Hammer', 'Bullish_Engulfing', 'Bearish_Engulfing',
        'SMA_50', 'SMA_200', 'Golden_Cross', 'Death_Cross',
        'BB_Breakout_Upper', 'BB_Breakout_Lower'
    ]

    train_size_idx = int(len(data) * 0.8)

    # Scaler fitted only on training portion to prevent data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[features].iloc[:train_size_idx].values)
    scaled_data = scaler.transform(data[features].values)

    def create_dataset(dataset, time_step=50):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), :])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 50
    X, y = create_dataset(scaled_data, time_step)

    train_size_seq = int(len(X) * 0.8)
    X_train, X_test = X[:train_size_seq], X[train_size_seq:]
    y_train, y_test = y[:train_size_seq], y[train_size_seq:]

    # Two LSTM layers with Dropout(0.2) to regularize; EarlyStopping restores best weights
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    loss_curve_fig = go.Figure()
    loss_curve_fig.add_trace(go.Scatter(
        x=list(range(1, len(history.history['loss']) + 1)),
        y=history.history['loss'],
        mode='lines', name='Training Loss', line=dict(color='#1f77b4')
    ))
    loss_curve_fig.add_trace(go.Scatter(
        x=list(range(1, len(history.history['val_loss']) + 1)),
        y=history.history['val_loss'],
        mode='lines', name='Validation Loss', line=dict(color='#ff7f0e')
    ))
    loss_curve_fig.update_layout(
        title="Multivariate LSTM Training Loss Curve",
        xaxis_title="Epoch", yaxis_title="Loss"
    )

    return model, history, loss_curve_fig, scaler, scaled_data[-time_step:]