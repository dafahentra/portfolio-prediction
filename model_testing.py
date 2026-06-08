"""
model_testing.py — Exploratory model comparison script (not part of the main app).

Compares classification and regression approaches for next-day price prediction
on a single stock before the multivariate LSTM pipeline was finalised.
Models tested: Random Forest, XGBoost, Linear Regression (regression), ARIMA.
"""

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

ticker = 'AAPL'
raw = yf.Ticker(ticker).history(period='5y')
raw = raw[['Close', 'Volume']]

# ---------------------------------------------------------------------------
# Classification target: 1 if next-day Close > today's Close, else 0
# ---------------------------------------------------------------------------

clf_data = raw.copy()
clf_data['Target'] = (clf_data['Close'].shift(-1) > clf_data['Close']).astype(int)
clf_data = clf_data.dropna()

X_clf = clf_data[['Close']]
y_clf = clf_data['Target']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, shuffle=False
)

# Classification models
# XGBClassifier: eval_metric is passed directly; use_label_encoder was removed in XGBoost >= 2.0
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":       XGBClassifier(eval_metric='logloss', random_state=42),
}

print("=== Classification Results ===")
for name, model in classifiers.items():
    model.fit(X_train_clf, y_train_clf)
    y_pred = model.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred)
    cm  = confusion_matrix(y_test_clf, y_pred)
    print(f"{name}  Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 40)

    plt.figure(figsize=(12, 4))
    plt.plot(y_test_clf.index, y_test_clf.values, label='Actual (1=Up, 0=Down)')
    plt.plot(y_test_clf.index, y_pred,            label=f'{name} Predictions', linestyle='dotted')
    plt.legend()
    plt.title(f'{ticker} Price Direction — {name}')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Regression target: next-day Close price
# ---------------------------------------------------------------------------

reg_data = raw.copy()
reg_data['Target'] = reg_data['Close'].shift(-1)
reg_data = reg_data.dropna()

X_reg = reg_data[['Close']]
y_reg = reg_data['Target']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, shuffle=False
)

# Linear Regression baseline
linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)
y_pred_lr = linear_model.predict(X_test_reg)

plt.figure(figsize=(12, 4))
plt.plot(y_test_reg.index, y_test_reg.values, label='Actual')
plt.plot(y_test_reg.index, y_pred_lr,         label='Linear Regression', linestyle='dashed')
plt.legend()
plt.title(f'{ticker} Price Prediction — Linear Regression')
plt.tight_layout()
plt.show()

# Ensemble regressors
regressors = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost":       XGBRegressor(n_estimators=100, random_state=42),
}

print("\n=== Regression Results ===")
for name, model in regressors.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred)
    print(f"{name}  MSE: {mse:.2f}")

    plt.figure(figsize=(12, 4))
    plt.plot(y_test_reg.index, y_test_reg.values, label='Actual')
    plt.plot(y_test_reg.index, y_pred,             label=f'{name}', linestyle='dotted')
    plt.legend()
    plt.title(f'{ticker} Price Prediction — {name}')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# ARIMA time-series baseline
# ---------------------------------------------------------------------------

series = raw['Close']
train_size = int(0.8 * len(series))
train_series, test_series = series.iloc[:train_size], series.iloc[train_size:]

arima_model  = ARIMA(train_series, order=(5, 1, 0))
arima_result = arima_model.fit()
arima_pred   = arima_result.forecast(steps=len(test_series))

plt.figure(figsize=(12, 4))
plt.plot(test_series.index, test_series.values, label='Actual')
plt.plot(test_series.index, arima_pred,         label='ARIMA (5,1,0)', linestyle='dotted')
plt.legend()
plt.title(f'{ticker} Price Prediction — ARIMA')
plt.tight_layout()
plt.show()