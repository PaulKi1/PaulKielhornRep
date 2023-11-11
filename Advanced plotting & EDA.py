import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from scipy.stats import skew, kurtosis
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# Note: this is a follow up on "Basic plotting"
# Most parameters should stay the same
# Some things here are completely irrelevant and were found on the internet.

df = pd.read_excel(r"C:")

# Calculate new things
df['option_mid'] = (df['bid'] + df['ask']) / 2
df['underlying_mid'] = (df['underlying_bid'] + df['underlying_ask']) / 2
df['moneyness'] = df['active_underlying_price'] / df['strike']
print(df[['option_mid','close']])

# Skewness and Kurtosis of the log returns.
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['skewness'] = df.groupby('expiration')['log_return'].transform(lambda x: skew(x.dropna()))
df['kurtosis'] = df.groupby('expiration')['log_return'].transform(lambda x: kurtosis(x.dropna()))
# Kurtosis describes how the variable, in this case the returns, compare to that of a normal distribution



# Regressionsanalyse
df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])     # Indexing and changing to correct integer
df.set_index('quote_datetime', inplace=True)

# Filter for a specific strike
option_example = df[df['strike'] == 300]    # Groupby or for loop can be used for more strikes

# Ensure there's enough data. This is optional if the data is already "explored" quite well
if len(option_example) >= 2:
    # Assuming daily data with no clear seasonality; otherwise, set an appropriate period. Is there a way to check that?
    decomposition = seasonal_decompose(option_example['close'], model='additive')
    decomposition.plot()
    plt.show()
else:
    print("Not enough data points for seasonal decomposition.")

# Pairplot to visualize the relationships between Variables
sns.pairplot(df[['delta', 'gamma', 'theta', 'vega', 'rho']], diag_kind='kde')
plt.show()

# Scatter plot matrix
# Note: pandas scatter_matrix does not support the 'hue' parameter.
scatter_matrix(df[['close', 'delta', 'gamma', 'theta', 'vega']], figsize=(10, 10), alpha=0.2, diagonal='kde')
plt.show()

# Correlation Analysis
# Calculate the rolling correlation of option prices with the underlying asset price
rolling_corr = df['vega'].rolling(window=30).corr(df['active_underlying_price'])
plt.figure(figsize=(14, 7))
rolling_corr.plot(title='Rolling Correlation of Option Close with Underlying Price')
plt.show()
# Rolling correlation is just correlation and how it changes over time, being calculated with values from window

# Median polish
# result = sm.robust.medpolish(df.to_numpy()) # doesn't work, command for medpolish is needed or mathematically
# print(result)

# Trimean
def trimean(data):
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    return (q1 + 2*q2 + q3) / 4

print(trimean(df['implied_volatility']))
# Trimean is an estimate of central tendency. Can have advantage compared to median

# Predictive Model (as part of EDA to uncover relationships)
# A simple model to predict option prices based on 'Greeks' and other features
X = df[['delta', 'gamma', 'theta', 'vega', 'rho', 'moneyness', 'option_mid']]
y = df['close']
# df['next_close'] = df['active_underlying_price'].shift(-1)
# df.dropna(inplace=True)  # Make this work, since we are trying to predict future value. Has to be checked
# y = df['next_close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100)
# model = LinearRegression()
# model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

# Evaluate the model
Mean_squared_error = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {Mean_squared_error}')

# Importances can show which inputs are the most valuable for the prediction. This can be incorporated into an ML or so
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.show()