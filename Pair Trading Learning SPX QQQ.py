import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt



# Trying out some Pair trading. This is mainly a template for backtesting or something equivalent.
# Backtest might be wrong. Entry treshholds seem weird. Check how it does exactly

start_date = "2003-08-09"
end_date = "2023-08-16"
spx_data = yf.download('^GSPC', start=start_date, end=end_date)
nasdaq_data = yf.download('^NDX', start=start_date, end=end_date)

spx_data['Percent Gain'] = spx_data['Close'].pct_change() * 100
nasdaq_data['Percent Gain'] = nasdaq_data['Close'].pct_change() * 100

print(spx_data)
data = pd.concat([spx_data['Percent Gain'], nasdaq_data['Percent Gain']], axis=1)
data.columns = ['SPX Percent Gain', 'Nasdaq Percent Gain']
data.dropna(inplace=True)

spread = data['SPX Percent Gain'] - data['Nasdaq Percent Gain']

entry_thresholds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # Corrected variable name
exit_threshold = 0.5  # Corrected variable name

plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
for entry_threshold in entry_thresholds:
    spread_mean = spread.mean()
    spread_std = spread.std()

    positions = []
    pnl = []

    for i in range(len(spread)):
        print(f"i: {i}, spread: {spread[i]}, SPX: {spx_data['Close'][i]}, Nasdaq: {nasdaq_data['Close'][i]}")
        take_position = False  # Flag to indicate whether to take a position

        if spread[i] > spread_mean + entry_threshold * spread_std:
            positions.append(1)                 # Solve why 1 is here and -1 in 42
            take_position = True  # Set the flag to take a position
        elif spread[i] < spread_mean - entry_threshold * spread_std:
            positions.append(-1)
            take_position = True  # Set the flag to take a position
        else:
            positions.append(0)

        if i > 0:
            if abs(spread[i]) < exit_threshold * spread_std:
                pnl.append(positions[i - 1] * (spread[i] - spread[i - 1])) # Close position
                #pnl.append(positions[i - 1] * (spread[i] - spread[i - 1]))
                positions[i] = 0 #reset position
            else:
                #pnl.append(0)  # No position
                pnl.append(positions[i] * (spread[i] - spread[i - 1])) # Continue position
        else:
            pnl.append(0)
            #pnl.append(positions[i - 1] * (spread[i] - spread[i - 1]))
            print(f"i: {i}, positions length: {len(positions)}, pnl length: {len(pnl)}")

    #results = pd.DataFrame({'Spread': spread, 'Position': positions, 'P&L': pnl})
    # Calculate cumulative P&L
    #results['Cumulative P&L'] = results['P&L'].cumsum()

    #cumulative_pnl = np.cumsum(pnl)
    #plt.plot(data.index, cumulative_pnl, label=f'Entry Threshold: {entry_threshold}')

    cumulative_pnl = pd.Series(pnl).cumsum()
    plt.plot(data.index, cumulative_pnl, label=f'Entry Threshold: {entry_threshold}')

plt.axhline(0, color='black', linestyle='--', label='Zero P&L')
plt.xlabel('Time')
plt.ylabel('Cumulative P&L')
plt.title('Intraday Pair Trading Strategy Performance')
plt.legend()
plt.grid(True)

# Second plot: Spread
plt.subplot(2, 1, 2)
plt.plot(data.index, spread, label='Spread')
plt.xlabel('Time')
plt.ylabel('Spread')
plt.title('Spread Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()