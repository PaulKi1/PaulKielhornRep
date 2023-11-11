import yfinance as yf
import statsmodels.api as sm
import numpy as np

# Testing for stationary data
# Muss überarbeitet werden

start_date = "2010-01-01"
end_date = "2020-01-01"

spx_data = yf.download('^NDX', start=start_date, end=end_date, progress=False)

spx_returns = spx_data['Close'].pct_change().dropna()
spx_returns_median = np.median(spx_returns)

adf_test = sm.tsa.adfuller(spx_returns)

print(spx_returns_median)
print("ADF Statistic:", adf_test[0])
print("p-value:", adf_test[1])
print("Critical Values:", adf_test[4])
print("Is the time series stationary?", adf_test[1] < 0.05)