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
    Wilder's RSI using exponentially weighted moving average.
    com = window - 1 produces alpha = 1/window, matching Wilder's original smoothing.
    Single source of truth — reused by add_price_action_features and
    predict_next_7_days to ensure a consistent RSI definition throughout.
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))


def sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Annualized Sharpe ratio.
    Converts the annual risk-free rate to a daily rate (/ 252) then
    annualizes the daily excess-return ratio by sqrt(252).
    Returns 0.0 when standard deviation is zero or undefined.
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    std = excess_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess_returns.mean() / std) * np.sqrt(252)


def sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    """
    Sortino ratio using the standard semi-deviation formula (Frank Sortino / CFA Institute).

    Downside deviation:
        sigma_d = sqrt( mean( min(R_i - MAR, 0)^2 ) )

    where MAR (Minimum Acceptable Return) defaults to 0 (daily).
    The numerator uses the daily risk-free rate to measure excess return.
    Returns np.nan when downside deviation is zero (no negative deviations observed).
    """
    daily_rf = risk_free_rate / 252
    downside_diff = np.minimum(returns - target_return, 0)
    downside_dev = np.sqrt(np.mean(downside_diff ** 2))
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    return ((returns.mean() - daily_rf) / downside_dev) * np.sqrt(252)



def _scale_feature(scaler, value, feature_idx):
    """
    Apply MinMaxScaler transform to a single feature value without constructing
    a full feature matrix.  Equivalent to calling scaler.transform for one
    feature at one point; clips the output to [0, 1].
    Used internally by predict_next_7_days to scale each updated indicator
    before feeding it back into the LSTM input window.
    """
    if np.isnan(value):
        return 0.0
    feature_range = scaler.data_max_[feature_idx] - scaler.data_min_[feature_idx]
    if feature_range == 0:
        return 0.0
    return float(np.clip(
        (value - scaler.data_min_[feature_idx]) / feature_range,
        0.0, 1.0
    ))


def predict_next_7_days(model_lstm, scaler, last_seq, last_real_close,
                        close_history, volume_mean):
    """
    Autoregressive 7-step forecast using business-day trading cadence.
    At each step the LSTM predicts the next scaled Close; all computable
    indicators are re-derived from the growing Close buffer before the
    next iteration so the model always sees internally consistent features.

    Updated each step:
        Close        — predicted by LSTM
        Volume       — historical mean (best available proxy for unknown future volume)
        RSI          — Wilder 14-period EWM on the rolling Close buffer
        MACD         — 12-period minus 26-period EMA on the rolling Close buffer
        SMA_50/200   — rolling means of the last 50 / 200 Close values
        Golden/Death Cross — SMA_50 vs SMA_200 crossover relative to the previous step
        BB Breakout  — 20-day ±2 standard-deviation Bollinger Band breakout flags

    Set to 0 (no-pattern) each step:
        Is_Doji, Is_Hammer, Bullish_Engulfing, Bearish_Engulfing
        These require OHLC data that does not exist for future days.
        Setting them to 0 (neutral / no-pattern) is the most defensible choice.

    Feature index map — must stay aligned with train_lstm_model_with_graphs:
        [0]  Close             [1]  Volume
        [2]  RSI               [3]  MACD
        [4]  Is_Doji           [5]  Is_Hammer
        [6]  Bullish_Engulfing [7]  Bearish_Engulfing
        [8]  SMA_50            [9]  SMA_200
        [10] Golden_Cross      [11] Death_Cross
        [12] BB_Breakout_Upper [13] BB_Breakout_Lower

    Args:
        model_lstm      : trained Keras LSTM model
        scaler          : fitted MinMaxScaler (clip=True) from training
        last_seq        : np.ndarray (time_step, n_features) — seed window
        last_real_close : float — last observed Close price in USD
        close_history   : list of at least 250 historical Close values
        volume_mean     : float — historical mean volume used as future proxy

    Returns:
        predictions : list of 7 predicted prices in USD
        behavior    : list of 7 strings ('Increase' or 'Decrease')
    """
    predictions = []
    behavior = []
    current_seq = last_seq.copy()
    current_close = last_real_close
    close_buf = list(close_history)

    # Initialise previous-day SMAs for crossover detection on the first predicted step
    buf_s = pd.Series(close_buf)
    prev_sma50 = float(buf_s.iloc[-50:].mean()) if len(buf_s) >= 50 else float(buf_s.mean())
    prev_sma200 = float(buf_s.iloc[-200:].mean()) if len(buf_s) >= 200 else float(buf_s.mean())

    # Scale the historical mean volume once; reused at every step
    scaled_vol = _scale_feature(scaler, volume_mean, 1)

    for _ in range(7):
        input_seq = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        predicted_scaled_close = float(model_lstm.predict(input_seq, verbose=0)[0][0])

        # Inverse-transform only the Close column (index 0).
        # MinMaxScaler operates per-feature independently, so padding the
        # other columns with zeros does not affect the Close inversion.
        pad = np.zeros((1, current_seq.shape[1]))
        pad[0, 0] = predicted_scaled_close
        predicted_price = float(scaler.inverse_transform(pad)[0][0])

        predictions.append(predicted_price)
        behavior.append("Increase" if predicted_price > current_close else "Decrease")
        current_close = predicted_price
        close_buf.append(predicted_price)

        cs = pd.Series(close_buf)

        # RSI — Wilder 14-period EWM; com=13 gives alpha = 1/14
        delta = cs.diff()
        gain = delta.where(delta > 0, 0.0)
        loss_s = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss_s.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        new_rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        # MACD — difference between 12-period and 26-period EMA
        new_macd = float(
            (cs.ewm(span=12, adjust=False).mean()
             - cs.ewm(span=26, adjust=False).mean()).iloc[-1]
        )

        # Simple Moving Averages over the last 50 / 200 Close values in the buffer
        new_sma50 = float(cs.iloc[-50:].mean()) if len(cs) >= 50 else float(cs.mean())
        new_sma200 = float(cs.iloc[-200:].mean()) if len(cs) >= 200 else float(cs.mean())

        # Golden / Death Cross — detect SMA_50 / SMA_200 crossover vs. the previous step
        new_golden = 1.0 if (new_sma50 > new_sma200 and prev_sma50 <= prev_sma200) else 0.0
        new_death = 1.0 if (new_sma50 < new_sma200 and prev_sma50 >= prev_sma200) else 0.0
        prev_sma50, prev_sma200 = new_sma50, new_sma200

        # Bollinger Bands — 20-day rolling mean ± 2 standard deviations
        sma20 = float(cs.iloc[-20:].mean()) if len(cs) >= 20 else float(cs.mean())
        std20 = float(cs.iloc[-20:].std()) if len(cs) >= 20 else float(cs.std())
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        new_bb_up = 1.0 if predicted_price > bb_upper else 0.0
        new_bb_lo = 1.0 if predicted_price < bb_lower else 0.0

        # Build the updated 14-feature row for the next LSTM step
        new_day = current_seq[-1].copy()
        new_day[0] = predicted_scaled_close
        new_day[1] = scaled_vol                              # Volume: historical mean proxy
        new_day[2] = _scale_feature(scaler, new_rsi, 2)
        new_day[3] = _scale_feature(scaler, new_macd, 3)
        new_day[4] = 0.0                                     # Is_Doji: no OHLC for future days
        new_day[5] = 0.0                                     # Is_Hammer: no OHLC for future days
        new_day[6] = 0.0                                     # Bullish_Engulfing: no OHLC for future days
        new_day[7] = 0.0                                     # Bearish_Engulfing: no OHLC for future days
        new_day[8] = _scale_feature(scaler, new_sma50, 8)
        new_day[9] = _scale_feature(scaler, new_sma200, 9)
        new_day[10] = new_golden                             # Binary 0/1; already in [0, 1]
        new_day[11] = new_death
        new_day[12] = new_bb_up
        new_day[13] = new_bb_lo

        current_seq = np.append(current_seq[1:], [new_day], axis=0)

    return predictions, behavior


def add_price_action_features(data):
    """
    Compute all 14 technical features used by both the Random Forest and LSTM models.
    RSI delegates to calculate_rsi() to keep the Wilder EWM logic in one place
    and avoid divergence between the training and inference implementations.
    """
    data = data.copy()

    # Wilder's RSI (14-period) — delegates to calculate_rsi for a consistent implementation
    data['RSI'] = calculate_rsi(data)

    # MACD — difference between 12-period and 26-period exponential moving average
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2

    # Candlestick pattern geometry
    body = abs(data['Close'] - data['Open'])
    upper_shadow = data['High'] - data[['Open', 'Close']].max(axis=1)
    lower_shadow = data[['Open', 'Close']].min(axis=1) - data['Low']
    avg_body = body.rolling(window=14).mean()

    # Doji: body is less than 10% of the 14-day average body size
    data['Is_Doji'] = (body < (0.1 * avg_body)).astype(int)

    # Hammer: small body with a long lower shadow (> 2× body) and a tiny upper shadow.
    # The body > 0.1 × avg_body guard prevents overlap with Doji classification.
    data['Is_Hammer'] = (
        (body > (avg_body * 0.1)) &
        (body < (0.3 * (data['High'] - data['Low']))) &
        (lower_shadow > (2 * body)) &
        (upper_shadow < (0.3 * body))
    ).astype(int)

    # Bullish Engulfing: prior bearish candle fully engulfed by the current bullish candle
    data['Bullish_Engulfing'] = (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Open'] < data['Close']) &
        (data['Open'] <= data['Close'].shift(1)) &
        (data['Close'] >= data['Open'].shift(1))
    ).astype(int)

    # Bearish Engulfing: prior bullish candle fully engulfed by the current bearish candle
    data['Bearish_Engulfing'] = (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Open'] > data['Close']) &
        (data['Open'] >= data['Close'].shift(1)) &
        (data['Close'] <= data['Open'].shift(1))
    ).astype(int)

    # 50-day and 200-day Simple Moving Averages for medium- and long-term trend context
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Golden Cross: SMA_50 crosses above SMA_200 — a long-term bullish signal
    data['Golden_Cross'] = (
        (data['SMA_50'] > data['SMA_200']) &
        (data['SMA_50'].shift(1) <= data['SMA_200'].shift(1))
    ).astype(int)

    # Death Cross: SMA_50 crosses below SMA_200 — a long-term bearish signal
    data['Death_Cross'] = (
        (data['SMA_50'] < data['SMA_200']) &
        (data['SMA_50'].shift(1) >= data['SMA_200'].shift(1))
    ).astype(int)

    # Bollinger Bands — 20-day rolling mean ± 2 standard deviations
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    std_20 = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (std_20 * 2)
    data['BB_Lower'] = data['SMA_20'] - (std_20 * 2)
    data['BB_Breakout_Upper'] = (data['Close'] > data['BB_Upper']).astype(int)
    data['BB_Breakout_Lower'] = (data['Close'] < data['BB_Lower']).astype(int)

    return data


def train_rf_model_with_graphs(data):
    """
    Train a Random Forest classifier for next-day price direction prediction.

    Cross-validation uses 5-fold TimeSeriesSplit applied only to the training
    portion (first 80%) so no test-period data leaks into the reported CV score.
    The confusion matrix and Precision-Recall curve come from the final
    holdout evaluation on the test portion (last 20%).
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

    # Temporal 80 / 20 split — training set is isolated before CV so that
    # later folds cannot validate on the held-out test period.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 5-fold TimeSeriesSplit CV runs exclusively on the training portion
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42
        )
        clf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        cv_scores.append(
            accuracy_score(y_train.iloc[val_idx], clf.predict(X_train.iloc[val_idx]))
        )

    # Final model trained on the full training set for holdout evaluation
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
    Train a multivariate LSTM on 14 price-action features with a 50-day lookback window.

    Scaler:
        Fitted on the training portion only (no data leakage).
        clip=True keeps test-period values — which may exceed the training price
        range for growth stocks — bounded to [0, 1] so the LSTM never receives
        out-of-distribution inputs during either evaluation or live prediction.

    Split boundary alignment:
        train_size_seq is derived from train_size_idx so that the scaler boundary
        and the sequence boundary coincide at the same row.  Every window in
        X_train has its label strictly inside the training portion; every window
        in X_test has its label in the test portion.

    Returns:
        model          — trained Keras LSTM
        history        — Keras History object
        loss_curve_fig — Plotly training / validation loss curve
        scaler         — MinMaxScaler(clip=True) fitted on the training portion only
        last_seq       — last 50-day scaled window (seed for 7-day forecast)
        test_metrics   — dict{'rmse': float, 'directional_accuracy': float}
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

    # Shared 80 / 20 temporal cut-point used by both the scaler and the sequence split
    time_step = 50
    train_size_idx = int(len(data) * 0.8)

    # Fit scaler on training rows only; clip=True clamps out-of-range test values
    # to [0, 1] instead of allowing them to exceed the training bounds
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    scaler.fit(data[features].iloc[:train_size_idx].values)
    scaled_data = scaler.transform(data[features].values)

    def create_dataset(dataset, time_step=50):
        """
        Build (X, y) window pairs from a scaled 2-D array.
        Window i uses rows [i, i + time_step) as features;
        the label is the scaled Close at row i + time_step.
        range(N - time_step) covers every valid index so that
        no labeled sample is skipped.
        """
        X, y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), :])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, time_step)

    # Align the sequence split with the scaler boundary.
    # Window i has its label at row (i + time_step), so windows with
    # i < (train_size_idx - time_step) have labels strictly in the training portion.
    train_size_seq = train_size_idx - time_step
    X_train, X_test = X[:train_size_seq], X[train_size_seq:]
    y_train, y_test = y[:train_size_seq], y[train_size_seq:]

    # Two stacked LSTM layers with Dropout(0.2) for regularisation;
    # EarlyStopping monitors val_loss and restores the best weights found
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

    # Evaluate on the test set in USD (original price scale) for interpretability.
    # Both predicted and actual scaled values are inverse-transformed via zero-padded
    # matrices so that only the Close column (index 0) is unscaled.
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    n_features = X.shape[2]

    pad_pred = np.zeros((len(y_pred_scaled), n_features))
    pad_pred[:, 0] = y_pred_scaled
    y_pred_prices = scaler.inverse_transform(pad_pred)[:, 0]

    pad_actual = np.zeros((len(y_test), n_features))
    pad_actual[:, 0] = y_test
    y_actual_prices = scaler.inverse_transform(pad_actual)[:, 0]

    # Note: y_actual_prices reflects clip=True scaled values. For growth stocks
    # where test-period prices exceed the training-period maximum, actuals are
    # capped at that training peak, which understates the true RMSE. This is
    # the accepted trade-off of fitting the scaler on training data only.
    rmse = float(np.sqrt(mean_squared_error(y_actual_prices, y_pred_prices)))
    dir_acc = float(
        np.mean(np.sign(np.diff(y_pred_prices)) == np.sign(np.diff(y_actual_prices)))
    )
    test_metrics = {'rmse': rmse, 'directional_accuracy': dir_acc}

    # Training / validation loss curve for sidebar display
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

    # last_seq: the final 50-row window of scaled data, used as the seed
    # for the 7-day autoregressive forecast in predict_next_7_days()
    return model, history, loss_curve_fig, scaler, scaled_data[-time_step:], test_metrics