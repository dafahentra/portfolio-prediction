import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Fetch historical data for the stock
ticker = 'AAPL'
data = yf.Ticker(ticker).history(period='5y')

# Prepare the data for modeling
data = data[['Close']]

# Classification: Predict if price increases
data['Target_Classifier'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# Features and target for classification
X_class = data[['Close']]
y_class = data['Target_Classifier']

# Prepare regression target
data['Target_Regressor'] = data['Close'].shift(-1)
data = data.dropna()

X_reg = data[['Close']]
y_reg = data['Target_Regressor']

# Split data into training and testing sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, shuffle=False)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, shuffle=False)

# Initialize models
models_class = {
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

models_reg = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate classification models
for name, model in models_class.items():
    model.fit(X_train_class, y_train_class)
    y_pred = model.predict(X_test_class)
    accuracy = accuracy_score(y_test_class, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
    
    # Plot classification results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_class.index, y_test_class, label='Actual (Increase: 1, Decrease: 0)', linestyle='-')
    plt.plot(y_test_class.index, y_pred, label=f'{name} Predictions', linestyle='dotted')
    plt.legend()
    plt.title(f'{ticker} Stock Price Direction Prediction ({name})')
    plt.show()

# Train and evaluate regression models
for name, model in models_reg.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred)
    print(f"{name} Mean Squared Error: {mse:.2f}")
    
    # Plot regression results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_reg.index, y_test_reg, label='Actual')
    plt.plot(y_test_reg.index, y_pred, label=f'{name} Predictions', linestyle='dotted')
    plt.legend()
    plt.title(f'{ticker} Stock Price Prediction ({name})')
    plt.show()

# ARIMA model
train_size = int(0.8 * len(data))
train_data, test_data = data['Close'][:train_size], data['Close'][train_size:]

# Fit ARIMA model
arima_model = ARIMA(train_data, order=(5, 1, 0))  # Adjust (p, d, q) as needed
arima_result = arima_model.fit()
arima_pred = arima_result.forecast(steps=len(test_data))

# Plot ARIMA results
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Actual')
plt.plot(test_data.index, arima_pred, label='ARIMA Predictions', linestyle='dotted')
plt.legend()
plt.title(f'{ticker} Stock Price Prediction (ARIMA)')
plt.show()



#----------- Decided -------------------------

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Fetch historical data for the stock
ticker = 'AAPL'
data = yf.Ticker(ticker).history(period='5y')

# Prepare the data for classification
data = data[['Close']]
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if price increases, 0 otherwise
data = data.dropna()

# Features and target
X = data[['Close']]
y = data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Store performance
results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        "accuracy": accuracy,
        "confusion_matrix": cm
    }

# Plot actual vs predicted for Linear Regression
# Prepare the data for regression
data['Target'] = data['Close'].shift(-1)  # The target is the next day's closing price
data = data.dropna()

X_reg = data[['Close']]
y_reg = data['Target']

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, shuffle=False)

# Train regression model
linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = linear_model.predict(X_test_reg)

# Plot regression
plt.figure(figsize=(12, 6))
plt.plot(y_test_reg.index, y_test_reg, label='Actual')
plt.plot(y_test_reg.index, y_pred_reg, label='Predicted', linestyle='dashed')
plt.legend()
plt.title(f'{ticker} Stock Price Prediction (Linear Regression)')
plt.show()

# Print classification results
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("-" * 30)
