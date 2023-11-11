from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np

df = pd.read_excel(r"C:")
print(df.dtypes)
columns_for_features = ["gamma","vega","ask_size","ask","bid_size","bid"] # Put the columns that should be tried. Can be done in for loop if
# you want to fry the PC

# Add the shifted 'underlying_price' as the target variable. This makes it so the model will try to find the pattern for
# the price in the next period
df['next_underlying_price'] = df['active_underlying_price'].shift(-1)

# Drop the last row with NaN 'next_underlying_price'
df = df[:-1]

# Select features and target
X = df[columns_for_features]
y = df['next_underlying_price']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=min(len(columns_for_features), 10))  # Number of components should be appropriate
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize the model and train the model has to be updated with the big ML code
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Make predictions and calculate the error
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# Creating a test trading strategy
def trading_strategy(predictions, prices):
    buy_signals = []
    sell_signals = []

    # Start from the first index of the predictions
    for i in range(1, len(predictions)):  # Starting from 1 to avoid index error
        if predictions[i] > prices[i - 1]:
            buy_signals.append(i)
        elif predictions[i] < prices[i - 1]:
            sell_signals.append(i)

    return buy_signals, sell_signals

# Preparing the DataFrame for backtesting
df_test = pd.DataFrame(df.iloc[len(X_train):len(X_train) + len(y_pred)])
df_test['predicted_price'] = y_pred

buy_signals, sell_signals = trading_strategy(df_test['predicted_price'].values, df_test['active_underlying_price'].values)
print(buy_signals)
print(sell_signals)

buy_signal_indices = df_test.index[buy_signals]
sell_signal_indices = df_test.index[sell_signals]

df_test['buy_signal'] = 0
df_test['sell_signal'] = 0
df_test.loc[buy_signal_indices, 'buy_signal'] = 1
df_test.loc[sell_signal_indices, 'sell_signal'] = 1

new_df = df_test[['quote_datetime', 'active_underlying_price', 'buy_signal', 'sell_signal']]
new_df['quote_datetime'] = pd.to_datetime(new_df['quote_datetime'])
new_df = new_df.set_index('quote_datetime')
new_df.loc[:, 'Open'] = new_df.loc[:, 'Close'] = new_df.loc[:, 'High'] = new_df.loc[:, 'Low'] = new_df.loc[:, 'active_underlying_price']
new_df.loc[:, 'Volume'] = 1

# Custom Strategy for Backtesting
class CustomIndexStrategy(Strategy):
    def init(self):
        self.buy_signals = self.data.df['buy_signal']
        self.sell_signals = self.data.df['sell_signal']

    def next(self):
        if self.buy_signals[-1]:
            if self.position:
                self.position.close()  # Close existing position before a new buy
            self.buy()

        if self.sell_signals[-1]:
            if self.position:
                self.position.close()


# Running the backtest. Backtest doesn't really work with shortselling. Has to be changed to either exit position if short
# signal or other workaround
def run_backtest(Strategy, data):
    bt = Backtest(data, Strategy, cash=10000, commission=0)
    stats = bt.run()
    bt.plot()
    return stats

stats = run_backtest(CustomIndexStrategy, new_df)
print(stats)

