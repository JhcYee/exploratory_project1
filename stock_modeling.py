import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Make sure to import mdates
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = yf.download('SPY', start='2001-01-01', end='2024-09-30')
print(df.head())
# Use 'Close' as the feature and target. Shift target column by 1 to predict the next day's price.
df['Prediction'] = df['Close'].shift(-1)
# Remove the last row, as it will have a NaN in the 'Prediction' column
df = df[:-1]
# Add moving averages as features
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
# Drop rows with NaN values (from moving averages)
df = df.dropna()
# Features (you can add more, e.g., moving averages, volume, etc.)
X = np.array(df[['Close', 'SMA_50', 'SMA_200']])
# Target (next day's closing price)
y = np.array(df['Prediction'])
# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Check model coefficients
print("Model Coefficients: ", model.coef_)
# Predict on the test set
predictions = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error: ", mse)

# Plot the actual vs predicted prices
plt.figure(figsize=(10,6))
plt.plot(df.index[-len(y_test):], y_test, color='blue', label='Actual Prices')
plt.plot(df.index[-len(predictions):], predictions, color='red', label='Predicted Prices')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # '%b' for month, '%Y' for year
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

#--------------------------------------------------
from sklearn.model_selection import cross_val_score

# Perform cross-validation on the model
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-Validation Scores: ", cv_scores)