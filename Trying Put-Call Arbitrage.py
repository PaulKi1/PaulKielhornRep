import pandas as pd
import numpy as np

df = pd.read_excel(r"C:")

# Trying to check if Put Call Parity can be exploited for dataset
# Put Call Parity: C - P = S - K * exp(-r * T)

risk_free_rate = 0.05
arbitrage_treshold = 0.01
# Ensure that the data types are correct. This is the issue for a lot of different codes
df['expiration'] = pd.to_datetime(df['expiration'])
df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])

# Calculate time to expiration in years
df['T'] = (df['expiration'] - df['quote_datetime']).dt.days / 365

# Calculate the present value of the strike price aka discount. Task try to use the curve
df['K_present_value'] = df['strike'] * np.exp(-risk_free_rate * df['T'])

# Calculate left handed (LHS) and right handed side (RHS) of Put-Call Parity
df['LHS'] = df['call_price'] - df['put_price']
df['RHS'] = df['underlying_price'] - df['K_present_value']

# If difference is significant then there is an arbitrage opportunity
df['parity_difference'] = df['LHS'] - df['RHS']
df['arbitrage_opportunity'] = np.abs(df['parity_difference']) > arbitrage_treshold

# Filter the DataFrame for potential arbitrage opportunities
arbitrage_opportunities = df[df['arbitrage_opportunity']]

print(arbitrage_opportunities[['strike', 'call_price', 'put_price', 'underlying_price', 'parity_difference']])
