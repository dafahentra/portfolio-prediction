from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import pandas as pd


def format_price(value):
    return f"${round(value, 2)}"


def calculate_rsi(data, window=14):
    """
    Wilder's RSI using EWM; com = window-1 gives alpha = 1/window.
    Single source of truth — reused by add_price_action_features.
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))


def sharpe_ratio(returns, risk_free_rate=0.02):
    """Annualized Sharpe ratio with daily risk-free rate adjustment."""
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    std = excess_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess_returns.mean() / std) * np.sqrt(252)


def sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    """
    FIX #1 — Sortino Ratio: correct semi-deviation formula.

    Standard definition (Frank Sortino / CFA Institute):
        σ_d = sqrt( mean( min(R_i - MAR, 0)^2 ) )

    Previous implementation incorrectly used:
        std( {R_i | R_i < 0} )   ← wrong centering (mean of negatives, not MAR)
                                  ← wrong denominator (n_down-1, not N_total)
    """
    daily_rf = risk_free_rate / 252
    downside_diff = np.minimum(returns - target_return, 0)
    downside_dev = np.sqrt(np.mean(downside_diff ** 2))
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    return ((returns.mean() - daily_rf) / downside_dev) * np.sqrt(252)


def track_portfolio(portfolio_tickers):
    """
    FIX #8 — Batched fetch: one yf.download call for all tickers instead of N
    separate yf.Ticker(...).history() calls.

    Returns a DataFrame with ticker symbols as columns.
    """
    raw = yf.download(portfolio_tickers, period="10y", progress=False)
    close = raw['Close']
    # yf.download with a single-item list may collapse to a Series
    if isinstance(close, pd.Series):
        close = close.to_frame(name=portfolio_tickers[0])
    return close


def _scale_feature(scaler, value, feature_idx):
    """
    Manually apply MinMaxScaler transform for a single feature value.
    Equivalent to scaler.transform for one feature; clips to [0, 1].
    Used internally by predict_next_7_days.
    """
    if np.isnan(value):
        return 0.0
    feature_range = scaler.data_max_[feature_idx] - scaler.data_min_[feature_idx]
    if feature_range == 0:
        return 0.0
    return float(np.clip((value - scaler.data_min_[feature_idx]) / feature_range, 0.0, 1.0))


def predict_next_7_days(model_lstm, scaler, last_seq, last_real_close,
                        close_history, volume_mean):
    """
    FIX #2 & FIX #5 — Unified autoregressive 7-day forecast with properly
    updated features at each step (replaces two divergent implementations
    in single_stock_pred.py and portfolio_pred.py).

    What is updated each step:
        Close        — predicted by LSTM
        Volume       — historical mean (best proxy; no oracle for future volume)
        RSI          — recomputed from rolling Close buffer (Wilder 14-period EWM)
        MACD         — recomputed from rolling Close buffer (12/26 EMA diff)
        SMA_50/200   — recomputed from rolling Close buffer
        Golden/Death Cross — derived from updated SMAs vs. previous step's SMAs
        BB Breakout  — recomputed from rolling 20-day Close buffer

    What cannot be updated (set to 0 = no-pattern):
        Is_Doji, Is_Hammer, Bullish_Engulfing, Bearish_Engulfing
        → these require OHLC data that does not exist for future days.
          Setting them to 0 (neutral / no-pattern) is the most defensible choice.

    Feature index map (must match train_lstm_model_with_graphs):
        [0]  Close             [1]  Volume
        [2]  RSI               [3]  MACD
        [4]  Is_Doji           [5]  Is_Hammer
        [6]  Bullish_Engulfing [7]  Bearish_Engulfing
        [8]  SMA_50            [9]  SMA_200
        [10] Golden_Cross      [11] Death_Cross
        [12] BB_Breakout_Upper [13] BB_Breakout_Lower

    Args:
        model_lstm      : trained Keras LSTM model
        scaler          : fitted MinMaxScaler from training
        last_seq        : np.ndarray (time_step, n_features) — seed window
        last_real_close : float — last observed Close price
        close_history   : list of at least 250 historical Close values
        volume_mean     : float — historical mean volume (proxy for future)

    Returns:
        predictions : list of 7 predicted prices (USD)
        behavior    : list of 7 strings ('Increase' or 'Decrease')
    """
    predictions = []
    behavior = []
    current_seq = last_seq.copy()
    current_close = last_real_close
    close_buf = list(close_history)

    # Initialise previous-day SMAs for cross-detection on the first predicted day
    buf_s = pd.Series(close_buf)
    prev_sma50 = float(buf_s.iloc[-50:].mean()) if len(buf_s) >= 50 else float(buf_s.mean())
    prev_sma200 = float(buf_s.iloc[-200:].mean()) if len(buf_s) >= 200 else float(buf_s.mean())

    scaled_vol = _scale_feature(scaler, volume_mean, 1)

    for _ in range(7):
        input_seq = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        predicted_scaled_close = float(model_lstm.predict(input_seq, verbose=0)[0][0])

        # Inverse-transform Close only; zeros in other columns don't affect index 0
        pad = np.zeros((1, current_seq.shape[1]))
        pad[0, 0] = predicted_scaled_close
        predicted_price = float(scaler.inverse_transform(pad)[0][0])

        predictions.append(predicted_price)
        behavior.append("Increase" if predicted_price > current_close else "Decrease")
        current_close = predicted_price
        close_buf.append(predicted_price)

        cs = pd.Series(close_buf)

        # --- RSI (Wilder 14-period EWM) ---
        delta = cs.diff()
        gain = delta.where(delta > 0, 0.0)
        loss_s = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss_s.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        new_rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        # --- MACD (12/26 EMA diff) ---
        new_macd = float(
            (cs.ewm(span=12, adjust=False).mean()
             - cs.ewm(span=26, adjust=False).mean()).iloc[-1]
        )

        # --- SMA 50 / 200 ---
        new_sma50 = float(cs.iloc[-50:].mean()) if len(cs) >= 50 else float(cs.mean())
        new_sma200 = float(cs.iloc[-200:].mean()) if len(cs) >= 200 else float(cs.mean())

        # --- Golden / Death Cross (crossover vs. previous step) ---
        new_golden = 1.0 if (new_sma50 > new_sma200 and prev_sma50 <= prev_sma200) else 0.0
        new_death = 1.0 if (new_sma50 < new_sma200 and prev_sma50 >= prev_sma200) else 0.0
        prev_sma50, prev_sma200 = new_sma50, new_sma200

        # --- Bollinger Bands (20-day, ±2σ) ---
        sma20 = float(cs.iloc[-20:].mean()) if len(cs) >= 20 else float(cs.mean())
        std20 = float(cs.iloc[-20:].std()) if len(cs) >= 20 else float(cs.std())
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        new_bb_up = 1.0 if predicted_price > bb_upper else 0.0
        new_bb_lo = 1.0 if predicted_price < bb_lower else 0.0

        # --- Build updated feature row ---
        new_day = current_seq[-1].copy()
        new_day[0] = predicted_scaled_close
        new_day[1] = scaled_vol                              # Volume: historical mean
        new_day[2] = _scale_feature(scaler, new_rsi, 2)
        new_day[3] = _scale_feature(scaler, new_macd, 3)
        new_day[4] = 0.0                                     # Is_Doji: no OHLC for future
        new_day[5] = 0.0                                     # Is_Hammer
        new_day[6] = 0.0                                     # Bullish_Engulfing
        new_day[7] = 0.0                                     # Bearish_Engulfing
        new_day[8] = _scale_feature(scaler, new_sma50, 8)
        new_day[9] = _scale_feature(scaler, new_sma200, 9)
        new_day[10] = new_golden                             # binary; 0/1 already in [0,1]
        new_day[11] = new_death
        new_day[12] = new_bb_up
        new_day[13] = new_bb_lo

        current_seq = np.append(current_seq[1:], [new_day], axis=0)

    return predictions, behavior


def add_price_action_features(data):
    """
    Compute all 14 technical features used by both RF and LSTM models.

    FIX #7 — RSI is now computed by calling calculate_rsi() instead of
    duplicating the same Wilder's EWM logic inline.
    """
    data = data.copy()

    # FIX #7: delegate to calculate_rsi — eliminates the duplicated logic
    data['RSI'] = calculate_rsi(data)

    # MACD (12/26 EMA difference)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2

    # Candlestick pattern flags
    body = abs(data['Close'] - data['Open'])
    upper_shadow = data['High'] - data[['Open', 'Close']].max(axis=1)
    lower_shadow = data[['Open', 'Close']].min(axis=1) - data['Low']
    avg_body = body.rolling(window=14).mean()

    data['Is_Doji'] = (body < (0.1 * avg_body)).astype(int)

    # Guard (body > avg_body * 0.1) prevents Doji from being flagged as Hammer
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
    """
    Train a Random Forest classifier for next-day direction prediction.
    CV accuracy uses 5-fold TimeSeriesSplit; confusion matrix and PR curve
    come from the final 80/20 temporal split model.
    """
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

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42
        )
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        cv_scores.append(accuracy_score(y.iloc[val_idx], clf.predict(X.iloc[val_idx])))

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
    """
    Train a multivariate LSTM on 14 price-action features with 50-day lookback.

    FIX #4 — Off-by-one in create_dataset corrected:
        range(N - time_step - 1) → range(N - time_step)
        Previous code wasted the last data point as a training target.

    FIX #9 — Returns test_metrics dict with meaningful evaluation in USD:
        'rmse'                : RMSE of test-set Close predictions in USD
        'directional_accuracy': fraction of correctly predicted price directions

    Returns:
        model         — trained Keras LSTM
        history       — Keras History object
        loss_curve_fig — Plotly training/validation loss curve
        scaler        — MinMaxScaler fitted on training portion only (no leakage)
        last_seq      — last 50-day scaled window (seed for 7-day forecast)
        test_metrics  — dict{'rmse': float, 'directional_accuracy': float}
    """
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

    # Scaler fitted only on training portion — no data leakage into test period
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[features].iloc[:train_size_idx].values)
    scaled_data = scaler.transform(data[features].values)

    def create_dataset(dataset, time_step=50):
        """
        FIX #4: range(N - time_step) instead of range(N - time_step - 1).
        The previous '-1' caused the last row to never be used as a y target,
        wasting one training sample per fit.
        """
        X, y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), :])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 50
    X, y = create_dataset(scaled_data, time_step)

    train_size_seq = int(len(X) * 0.8)
    X_train, X_test = X[:train_size_seq], X[train_size_seq:]
    y_train, y_test = y[:train_size_seq], y[train_size_seq:]

    # Two LSTM layers + Dropout(0.2); EarlyStopping restores best weights
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

    # --- FIX #9: Evaluate on test set in USD (meaningful scale) ---
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    n_features = X.shape[2]

    pad_pred = np.zeros((len(y_pred_scaled), n_features))
    pad_pred[:, 0] = y_pred_scaled
    y_pred_prices = scaler.inverse_transform(pad_pred)[:, 0]

    pad_actual = np.zeros((len(y_test), n_features))
    pad_actual[:, 0] = y_test
    y_actual_prices = scaler.inverse_transform(pad_actual)[:, 0]

    rmse = float(np.sqrt(mean_squared_error(y_actual_prices, y_pred_prices)))
    dir_acc = float(
        np.mean(np.sign(np.diff(y_pred_prices)) == np.sign(np.diff(y_actual_prices)))
    )
    test_metrics = {'rmse': rmse, 'directional_accuracy': dir_acc}

    # --- Loss curve ---
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

    return model, history, loss_curve_fig, scaler, scaled_data[-time_step:], test_metrics