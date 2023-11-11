import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_excel(r"C:")

# Extract relevant columns
expiration = df['Days till exp']
strike = df['strike']

# Create dataset for each variable (implied_volatility, Volume, bid_ask_spread)
# Data can be reshaped into grid (strike, expiration) for plotting
unique_strikes = np.unique(strike)
unique_expirations = np.unique(expiration)

# Define X,Y (strike, expiration values)
X = np.meshgrid(unique_strikes, unique_expirations)[0]
Y = np.meshgrid(unique_strikes, unique_expirations)[1]

# Create arrays for each variable
Z_implied_volatility = np.zeros_like(X)
Z_volume = np.zeros_like(X)
Z_bid_ask_spread = np.zeros_like(X)

for i, exp in enumerate(unique_expirations):
    for j, strk in enumerate(unique_strikes):
        # Assuming data contains matching values for expiration, implied volatility, volume, and bid-ask spread
        # Logic may need to be adjusted depending on your data format
        matching_row = df[(df['Days till exp'] == exp) & (df['strike'] == strk)]

        if not matching_row.empty:
            Z_implied_volatility[i, j] = float(matching_row['implied_volatility'].iloc[0])
            Z_volume[i, j] = float(matching_row['trade_volume'].iloc[0])                    #bid_ask_spread
            Z_bid_ask_spread[i, j] = float(matching_row['gamma'].iloc[0])

# Show vol surface
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot implied_volatility
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_implied_volatility, cmap='viridis')
ax1.set_xlabel('Strike')
ax1.set_ylabel('Days till exp')
ax1.set_zlabel('implied_volatility')
ax1.set_title('Implied Volatility Surface')

# Plot Volume
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_volume, cmap='viridis')
ax2.set_xlabel('Strike')
ax2.set_ylabel('Days till exp')
ax2.set_zlabel('Trade Volume')
ax2.set_title('Volume Surface')

# Plot Bid-Ask Spread
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z_bid_ask_spread, cmap='viridis')
ax3.set_xlabel('Strike')
ax3.set_ylabel('Days till exp')
ax3.set_zlabel('Bid-Ask Spread')
ax3.set_title('Bid-Ask Spread Surface')

plt.tight_layout()
plt.show()
