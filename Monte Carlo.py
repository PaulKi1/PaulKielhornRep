import numpy as np
import matplotlib.pyplot as plt


# Monte carlo with geometric brownian
# For future note, the :param and :return don't have to be there for the code to work
# To do: add correlation in here
def option_pricing_monte_carlo(S0, K, T, r, sigma, cash_payout, num_paths=10000, num_time_steps=100):
    """

    :param S0: Underlying price in t0
    :param K: Strike
    :param T: Time to maturity (in years)
    :param r: Risk-free interest rate (annual)
    :param cash_payout: The fixed cash amount paid if the option is ITM at maturity
    :param option_type: Type of option used
    :param sigma: Volatility of the underlying asset
    :param num_paths: Number of price paths to simulate
    :param num_time_steps: Number of time steps to simulate
    :return: Estimated price of the call option and simulated paths
    """
    dt = T / num_time_steps  # Time step size
    price_paths = np.zeros((num_paths, num_time_steps + 1))
    price_paths[:, 0] = S0

    # Generate paths
    for t in range(1, num_time_steps + 1):
        z = np.random.standard_normal(num_paths)  # Random standard normal distribution
        price_paths[:, t] = price_paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    # Calculate the payoff for each path at maturity. Note Calls can be turned into puts here, by rearanging the formulas

    # payoffs = np.maximum(price_paths[:, -1] - K, 0) # European Call

    # payoffs = np.maximum(K - price_paths[:, -1], 0) # European Put

    # average_price = np.mean(price_paths[:, 1:], axis=1) # Asian Call
    # payoffs = np.maximum(average_price - K, 0)

    # is_alive = np.all(price_paths < barrier_level, axis=1) # Up and out barrier call
    # payoffs = np.maximum(price_paths[:, -1] - K, 0) * is_alive

    # max_price = np.max(price_paths[:, 1:], axis=1)    # Lookback Call
    # payoffs = max_price - K


    # This is for a european call. Code has to be changed to always include the check for option type
    # Idea is to just have to change the payoff part
    #if option_type == 'call':
    #    payoffs = np.maximum(price_paths[:, -1] - K, 0)
    #else:
    #    payoffs = np.maximum(K - price_paths[:, -1], 0)



    in_the_money = price_paths[:, -1] > K   # Binary Option
    payoffs = np.where(in_the_money, cash_payout, 0) # The code checks if option is itm at maturity. Then payoff gets
    # calculated


    # Discount the payoffs back to present value
    pv_payoffs = np.exp(-r * T) * payoffs

    # Calculate the option price
    option_price = np.mean(pv_payoffs)

    return option_price, price_paths


# Example european call
# Underlying (S0): $100
# Strike (K): $105
# Time to maturity (T): 1 year
# Risk free rate (r): 5%
# Vol (sigma): 20%
# Number of paths: 10,000
# Number of time steps: 100
option_price_call, price_paths = option_pricing_monte_carlo(S0=100, K=105, T=1, r=0.05, sigma=0.2, cash_payout=10)
option_price_put, price_paths = option_pricing_monte_carlo(S0=100, K=105, T=1, r=0.05, sigma=0.2, cash_payout=10)

print(f"The estimated European call option price is: {option_price_call}")


# Visualizing the paths
plt.figure(figsize=(10, 6))
plt.plot(price_paths[:100].T)  # Plots first 10 graphs for readability. Changeable
plt.title('Simulated Price Paths for a European Call Option')
plt.xlabel('Time Step')
plt.ylabel('Stock Price')
plt.show()
