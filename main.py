import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Download recent historical stock data (e.g., last 60 days)
ticker = 'AAPL'
data = yf.download(ticker, period='60d', interval='1d', auto_adjust=False)

# Flatten MultiIndex columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# Use capitalized column names
ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
if not all(col in data.columns for col in ohlc_cols):
    ohlc_cols = [col.lower() for col in ohlc_cols]

# Drop rows with missing values before plotting
data = data.dropna(subset=ohlc_cols)

# 2. Plot candlestick chart
mpf.plot(data, type='candle', volume=True, style='yahoo', title=f'{ticker} Candlestick Chart')

# 3. Prepare data for ML
data['Target'] = data['Close'].shift(-1)  # Predict next day's close
data = data.dropna()

features = data[ohlc_cols]
target = data['Target']

# 4. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=False
)

# 5. Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Plot actual vs predicted closing prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Closing Price')
plt.plot(predictions, label='Predicted Closing Price')
plt.title(f'{ticker} Actual vs Predicted Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# 8. Predict the next closing price using the most recent data
latest_features = data[ohlc_cols].iloc[[-1]]  # Get the last row as DataFrame
suggested_price = model.predict(latest_features)[0]
print(f"\nSuggested next closing price for {ticker}: {suggested_price:.2f}")

# 9. Suggest Buy, Sell, or Hold
current_price = data['Close'].iloc[-1]
threshold = 0.01  # 1% threshold

if suggested_price > current_price * (1 + threshold):
    advice = "BUY"
elif suggested_price < current_price * (1 - threshold):
    advice = "SELL"
else:
    advice = "HOLD"

print(f"Current closing price: {current_price:.2f}")
print(f"Suggested action: {advice}")
