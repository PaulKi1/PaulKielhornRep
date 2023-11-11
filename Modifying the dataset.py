import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

df = pd.read_excel(r"c:")

df = df[df['option_type'] == 'C'] # To change the Optiontypes that get included
print(df['option_type'].count())


# df = pd.concat(data_frames, ignore_index=True) This was needed when combining multiple excels and not using MySQL

# This changes integer type of quote_datetime to be datetime64[ns]
df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
print(df.dtypes)


# The following part loads interest rate data and assigns it to the right rows to match the date provided in the excel
# This could be improved/made quicker
daily_rates = pd.read_excel(r'c:')
daily_rates['quote_datetime'] = pd.to_datetime(daily_rates['quote_datetime'])
rates_dictionary = daily_rates.set_index('quote_datetime')['rates'].to_dict()


df['part_date'] = df['quote_datetime'].dt.date
df['part_date'] = pd.to_datetime(df['part_date'])   # Not sure if this is necessary


column_to_check = 'part_date'

column_to_update = 'rates'

df['rates'] = None

for index, part_date in enumerate(df[column_to_check]):
    if part_date in rates_dictionary:
        df.at[index, 'rates'] = rates_dictionary[part_date]


# The following is to calculate  extra Variables
df['bid_ask_spread'] = df.apply(lambda row: (row['ask'] - row['bid'])/row['ask'] if row['bid'] > 0 and row['ask'] > 0 else 0, axis=1) # This is just the spread, not in %

df['Moneyness'] = df.apply(lambda row: ((row['active_underlying_price'])/row['strike'])/2,axis=1)

df['bid_ask_size_diff'] = df.apply(lambda row: abs(row['bid_size'] - row['ask_size']),axis=1) # Ebenfalls nur die absolute Size differenz und nicht prozentual


df = df[(df['Moneyness'] >= -1) & (df['Moneyness'] <= 1)] # Hiermit werden nur knapp ITM/OTM Options noch mit rein genommen
variables = ['quote_datetime', 'Moneyness', 'ask_size', 'bid_size', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'bid_ask_spread', 'bid_ask_size_diff', 'trade_volume']  #'open_interest', 'trade_volume'
combinations_variables = list(combinations(variables, 2))


# Hier werden alle möglichen Kombinationen geplotted. Andere Plotarten können ebenfalls verwendet werden
num_plots = len(combinations_variables)
fig, axes = plt.subplots(num_plots, figsize=(10, 8 * num_plots))

for i, (var1, var2) in enumerate(combinations_variables):
    x = df[var1]
    y = df[var2]

    plt.figure()
    ax = plt.gca()

    ax.scatter(x, y, s=2)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)

    ax.set_title(f'Scatter Plot: {var1} vs {var2}')

plt.tight_layout()

plt.show()

