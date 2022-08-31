import esig
import matplotlib.pyplot as plt
import iisignature
from src.HestonModel import generate_heston_paths

kappa = 3
theta = 0.02
v_0 = 0.02
sigma = 0.6
r = 0.04
S = 100
paths = 10000
steps = 10000
T = 1
rho = 0.75
prices: object
prices, sigs = generate_heston_paths(S, T, r, kappa, theta, v_0, rho, sigma, steps, paths, return_vol=True)

# Plot Heston Price paths
plt.figure(figsize=(8, 8))
plt.plot(prices.T)
plt.title('Heston Price Paths Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()
# Plot Heston volatility paths
plt.figure(figsize=(8, 8))
plt.plot(np.sqrt(sigs).T)
plt.title('Heston Stochastic Vol Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Volatility')
plt.show()

########################################################################################
time_array = np.arange(steps)
prices_with_time = np.zeros((paths, 2, steps))
prices_with_time[0]

for i in range(paths):
    prices_with_time[i] = np.array([prices[i], time_array])

# prices.shape
# Calculations of signature using iisignature package
sigs_iisignature = np.array([iisignature.sig(path.T, 5) for path in prices_with_time])

# Calculations of signature using esig package
sigs_esig = np.array([esig.stream2sig(path.T, 5) for path in prices_with_time])
print(esig.sigkeys(prices_with_time[0].T.shape[1], 5))

sigs_esig.shape
sigs_iisignature.shape
# Calculation of Expected signature component-wise
Exp_sign_iisignature = np.mean(sigs_iisignature, axis=0)
Exp_sign_esig = np.mean(sigs_esig, axis=0)
