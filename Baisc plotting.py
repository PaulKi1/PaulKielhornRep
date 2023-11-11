import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_excel (r"C:")

# Data cleaning and type conversion
# This converts all non-convertible values to NaN
#df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
# NaN values after conversion can be either dropped:
#df = df.dropna(subset=['strike'])
# Or fill them with a central tendency measure (mean, median). Dropping them/not using it at all is better in the case
# of this dataset
# df['strike'] = options_df['strike'].fillna(value=df['strike'].median())

# Display the first few rows to get a feel for the data
print(df.head())

# Summary statistics for numerical features
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Histograms for numerical features to understand distributions
df.hist(bins=len(df.columns), figsize=(20,15))
plt.tight_layout() # Adjust layout to prevent clipping of tick-labels. Still doesn't work not bothered
plt.show()

# Line plot for variable vs time. Can be made into loop to show all eaiser
plt.plot(df['quote_datetime'], df['delta'])
plt.xlabel('quote_datetime')
plt.ylabel('delta')
plt.title('Delta over time')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()

# Histogram
plt.hist(df['gamma'], bins=20)
plt.xlabel('gamma')
plt.ylabel('Frequency')
plt.title('Histogram of gamma')
plt.show()

# Box plots
numerical_cols = ['Days till exp', 'strike', 'open', 'high', 'low', 'close', 'trade_volume', 'bid_size', 'bid', 'ask_size', 'ask', 'underlying_bid', 'underlying_ask', 'active_underlying_price', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'open_interest']
for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Box plot of {col}')
    plt.show()

# Count plot
sns.countplot(x='option_type', data=df)
plt.title('Count of Option Types')
plt.show()

# Time series plot for option prices
plt.figure(figsize=(14, 7))
for option_type in df['option_type'].unique():
    subset = df[df['option_type'] == option_type]
    subset.groupby('quote_datetime')['close'].mean().plot(label=f'{option_type} close')
plt.legend()
plt.title('Option Close Price Over Time')
plt.show()

# Correlation heatmap
numerical_df = df.select_dtypes(include=[np.number])    # This has to be done, due to Option_type column being a problem
corr = numerical_df.corr()
plt.figure(figsize=(16, 10))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Implied volatility over time for different strikes
calls = df[df['option_type'] == 'C']
for strike in calls['strike'].unique():
    subset = calls[calls['strike'] == strike]
    subset.groupby('quote_datetime')['implied_volatility'].mean().plot(label=f'Strike {strike}')
plt.legend()
plt.title('Implied Volatility Over Time by Strike')
plt.show()

# Scatter plot of Delta vs Underlying Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='active_underlying_price', y='delta', hue='option_type', data=df)
plt.title('Delta vs Underlying Price')
plt.show()

# Horizon Graph
def plot_horizon(df, column, bands=2):
    horizon = df.copy()
    for i in range(1, bands + 1):
        horizon[f'band_{i}'] = np.clip(horizon[column] - (i - 1) * horizon[column].std(), 0, horizon[column].std())
        horizon[f'band_{-i}'] = -np.clip(horizon[column] + (i - 1) * horizon[column].std(), -horizon[column].std(), 0)

        plt.fill_between(horizon.index, horizon[f'band_{i}'], (i - 1) * horizon[column].std(), alpha=0.5)
        plt.fill_between(horizon.index, horizon[f'band_{-i}'], -(i - 1) * horizon[column].std(), alpha=0.5)

    plt.show()

plot_horizon(df, 'implied_volatility')
plt.title('Horizon Graph')

# Vol-table. Some values don't really work
vol_surface = df.pivot_table(index='Days till exp', columns='strike', values='implied_volatility')
plt.figure(figsize=(12, 8))
sns.heatmap(vol_surface, cmap='viridis')
plt.title('Implied Volatility Surface')
plt.xlabel('Strike Price')
plt.ylabel('Days to Expiration')
plt.show()
