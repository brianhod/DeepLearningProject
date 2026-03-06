import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mplfinance as mpf
# Download S&P 500 data
ticker = '^GSPC'
data = yf.download(ticker, start="2020-10-01", end="2024-11-06")

# Ensure the Index is in DateTime Format
# ensures dates are in a uniform DateTime format,
# which allows for easy time-based operations
data.index = pd.to_datetime(data.index)

# Select features and target
features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
target = data[['Close']]

# Scale features
feature_scaler = MinMaxScaler()
scaled_features = feature_scaler.fit_transform(features)

# Scale target
target_scaler = MinMaxScaler()
scaled_target = target_scaler.fit_transform(target)

# Function to create sequences
# Create sequences of data for time-series or
# sequential modeling (like with LSTM models).
# It helps organize data into sliding windows,
# which is often necessary when preparing data for models that require sequence inputs, such as RNNs, LSTMs
def create_sequences(features, target, time_step=1):
    # X this list will store the sequences of input data.
    # y: This list will store the corresponding target values for each sequence in X.
    X, y = [], []

  #This loop iterates through the data to create sequences of length time_step.
   # subtract time_step from the length of features to ensure we don’t go out of bounds.
    for i in range(len(features) - time_step):
        X.append(features[i:(i + time_step)]) #Takes a slice of the features array from index i to i + time_step. This slice represents a sequence of time_step elements.
        y.append(target[i + time_step]) # Takes the target value at the position right after the current sequence (index i + time_step). This is the value we want to predict for that sequence.
    return np.array(X), np.array(y)
# the lists X and y are converted into Numpy arrays, which are easier to work with in machine learning frameworks.

time_step = 100 # time_step is set to 100, which means each sequence will contain 100 consecutive time steps of data.

X, y = create_sequences(scaled_features, scaled_target, time_step)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
model = keras.models.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])),
    keras.layers.LSTM(50, return_sequences=False),
    keras.layers.Dense(25),
    keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, batch_size=1, epochs=3)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = target_scaler.inverse_transform(train_predict)
test_predict = target_scaler.inverse_transform(test_predict)

# Inverse transform actual values
y_train_actual = target_scaler.inverse_transform(y_train)
y_test_actual = target_scaler.inverse_transform(y_test)

r2_train = r2_score(y_train_actual, train_predict)
print(f'R2 Score for Training Data: {r2_train:.4f}')

# Calculate R2 score for testing data
r2_test = r2_score(y_test_actual, test_predict)
print(f'R2 Score for Testing Data: {r2_test:.4f}')


# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Original Data')

# Plot the closing price history of the stock
plt.figure(figsize=(16,8))
plt.title('Close Price History', fontweight='bold', fontsize=20)
plt.plot(data['Close'], color='green')
plt.xlabel('Date', fontsize=18, fontweight='bold')
plt.ylabel('Close Price ($)', fontsize=18, fontweight='bold')
plt.show()

mpf.plot(data, type='candle', style='charles', title='S&P 500 Candlestick Chart',
         ylabel='Price', volume=True)


close_sp500 = data[['Close']]
close_sp500_arr = close_sp500.values

ma100 = data.Close.rolling(100).mean()
ma100

plt.figure(figsize = (12,6))
plt.plot(data.Close)
plt.plot(ma100, 'r')
plt.title('Graph Of Moving Averages Of 100 Days')
plt.show()


ma200 = data.Close.rolling(200).mean()
ma200



plt.figure(figsize = (12,6))
plt.plot(data.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.title('Comparision Of 100 Days And 200 Days Moving Averages')
plt.show()





# Train predictions
train_plot = np.empty_like(data['Close'])
train_plot[:] = np.nan
train_plot[time_step:len(train_predict) + time_step] = train_predict.flatten()
plt.plot(data.index, train_plot, label='Train Predict')

# Test predictions
test_plot = np.empty_like(data['Close'])
test_plot[:] = np.nan
test_start_index = len(train_predict) + (time_step)
test_plot[test_start_index:test_start_index + len(test_predict)] = test_predict.flatten()
plt.plot(data.index, test_plot, label='Test Predict')

plt.xlabel("Date")
plt.ylabel("Price")
plt.suptitle('S&P 500 Price Prediction using LSTM')

plt.legend()
plt.show()




# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Original Data')

# Train predictions
train_plot = np.empty_like(data['Close'])
train_plot[:] = np.nan
train_plot[time_step:len(train_predict) + time_step] = train_predict.flatten()
plt.plot(data.index, train_plot, label='Train Predict')

# Test predictions
test_plot = np.empty_like(data['Close'])
test_plot[:] = np.nan
test_start_index = len(train_predict) + time_step
test_plot[test_start_index:test_start_index + len(test_predict)] = test_predict.flatten()
plt.plot(data.index, test_plot, label='Test Predict')

# Set x-axis to display by month and rotate labels
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45,  fontsize=8)

plt.xlabel("Date")
plt.ylabel("Price")
plt.suptitle('S&P 500 Price Prediction using LSTM')
plt.legend()
plt.grid(True)  # Optional: Add grid for easier visual interpretation
plt.show()



