
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# import iisignature
# import esig

# use Euler-Maruyama Discretisation  for Heston Model SDEs
def generate_heston_paths(start_price_S, maturity_T, r, kappa, theta, v_0, rho, sigma, steps, number_paths, return_vol=False):
    size = (number_paths, steps)
    sigs = np.zeros(size)
    prices = np.zeros(size)
    v_t = v_0
    dt = maturity_T / steps
    S_t = start_price_S
    for t in range(steps):
        # simulate stochastic processes
        WT = np.random.multivariate_normal(np.array([0, 0]), cov=np.array([[1, rho], [rho, 1]]), size=number_paths) * np.sqrt(
            dt)
        # generate stock prices
        S_t = S_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0]))
        # generate volatility
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + sigma * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = S_t
        sigs[:, t] = v_t

    if return_vol:
        return prices, sigs
    return prices

# Initial parameters
kappa = 2
theta = 0.15
v_0 = theta
rho = 0.6
sigma = 0.6
r = 0.04
S = 100
paths = 10000
steps = 100 #50 for 3d plots
T = 1


# prices, sigs = generate_heston_paths(S, T, r, kappa, theta, v_0, rho, sigma, steps, paths, return_vol=True)
# #
# prices.shape
# # Plot Heston Price paths
# plt.figure(figsize=(10, 6))
# plt.plot(prices.T)
# # plt.title('Heston Price Paths Simulation')
# plt.xlabel('Time t')
# plt.ylabel('Stock Price')
# plt.show()

# # Plot Heston volatility paths
# plt.figure(figsize=(8, 8))
# plt.plot(np.sqrt(sigs).T)
# plt.title('Heston Stochastic Vol Simulation')
# plt.xlabel('Time Steps')
# plt.ylabel('Volatility')
# plt.show()
