import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Download Nasdaq data from Yahoo Finance
start_date = "2020-01-01"
end_date = "2021-01-01"
nasdaq_data = yf.download('^NDX', start=start_date, end=end_date, progress=False)
# Create a DataFrame using Nasdaq data
df = pd.DataFrame({
    'Date': nasdaq_data.index,
    'Price': nasdaq_data['Adj Close']  # Use the Adjusted Close price for analysis
})
df.set_index('Date', inplace=True)

# Define the short-term and long-term moving averages
short_window = 10
long_window = 20

# Calculate moving averages
df['Short_MA'] = df['Price'].rolling(window=short_window).mean()
df['Long_MA'] = df['Price'].rolling(window=long_window).mean()

# Create a signal when the short-term MA crosses above the long-term MA
df['Signal'] = np.where(df['Short_MA'] > df['Long_MA'], 1, 0)
df['Position'] = df['Signal'].diff()  # Calculate position changes

# Backtesting
initial_balance = 100000  # Initial portfolio balance
balance = initial_balance
position = 0
buy_price = 0  # Store the buy price
profit_percent = 0.05  # 10% profit target

# Lists to store balance at each point and sell signals
balance_history = []
sell_signals = []

for i, row in df.iterrows():
    if row['Position'] == 1:  # Buy signal
        position = balance // row['Price']
        balance += position * row['Price']
        buy_price = row['Price']
    elif row['Position'] == -1:  # Sell signal
        while position > 0:
            balance += position * (row['Price'] - buy_price)  # Calculate profit/loss
            position = 0
            if balance >= (1 + profit_percent) * initial_balance:
                sell_signals.append(row.name)  # Record the time of selling for profit

    # Calculate total balance at each point
    total_balance = balance + position * (row['Price'] - buy_price)  # Balance with profit/loss
    balance_history.append(total_balance)

# Calculate final balance
final_balance = balance + position * (df['Price'].iloc[-1] - buy_price)

# Print results
print(f"Initial Balance: ${initial_balance:.2f}")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Total Return: {((final_balance - initial_balance) / initial_balance) * 100:.2f}%")

# Plot the strategy, balance history, and sell signals
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(df.index, df['Price'], label='Price')
plt.plot(df.index, df['Short_MA'], label=f'Short MA ({short_window} days)')
plt.plot(df.index, df['Long_MA'], label=f'Long MA ({long_window} days)')
plt.plot(df[df['Position'] == 1].index, df['Short_MA'][df['Position'] == 1], '^', markersize=10, color='g',
         label='Buy Signal')
plt.plot(df[df['Position'] == -1].index, df['Short_MA'][df['Position'] == -1], 'v', markersize=10, color='r',
         label='Sell Signal')
plt.legend()
plt.title('Moving Average Crossover Strategy')
plt.xlabel('Date')
plt.ylabel('Price')

plt.subplot(3, 1, 2)
plt.plot(df.index, balance_history, label='Portfolio Balance', color='b')
plt.scatter(sell_signals, [balance_history[df.index.get_loc(date)] for date in sell_signals], color='g',
            label='Sell for Profit')
plt.legend()
plt.title('Portfolio Balance Over Time')
plt.xlabel('Date')
plt.ylabel('Balance')

plt.subplot(3, 1, 3)
plt.plot(df.index, df['Position'], 'o', markersize=5, color='b', label='Position')
plt.legend()
plt.title('Position (1: Buy, -1: Sell)')
plt.xlabel('Date')
plt.ylabel('Position')
plt.tight_layout()
plt.show()
