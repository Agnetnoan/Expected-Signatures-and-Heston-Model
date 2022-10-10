from src.HestonModel import generate_heston_paths
import numpy as np
import esig
import matplotlib.pyplot as plt
import iisignature
import signatory
import torch
from tqdm.auto import tqdm

# constant Heston parameters
kappa = 3
theta = 0.02
v_0 = theta  #
rho = 0.75
sigma = 0.6
r = 0.04

S = 100
paths = 10000
steps = 100
T = 1

# Get Heston prices and volatilises
prices, sigs = generate_heston_paths(S, T, r, kappa, theta, v_0, rho, sigma, steps, paths, return_vol=True)

# Plot Heston Price paths
plt.figure(figsize=(8, 8))
plt.plot(prices.T)
plt.title('Heston Price Paths Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()
# # Plot Heston volatility paths
# plt.figure(figsize=(8, 8))
# plt.plot(np.sqrt(sigs).T)
# plt.title('Heston Stochastic Vol Simulation')
# plt.xlabel('Time Steps')
# plt.ylabel('Volatility')
# plt.show()
########################################################################################
time_array = np.arange(steps)
prices_with_time = np.zeros((paths, 2, steps))
prices_with_time[0]

for i in range(paths):
    prices_with_time[i] = np.array([prices[i], time_array])

prices.shape
# Calculations of signature using iisignature package
sign_iisignature = np.array([iisignature.sig(path.T, 3) for path in prices_with_time])

# Calculations of signature using esig package
sign_esig = np.array([esig.stream2sig(path.T, 3) for path in prices_with_time])
print(esig.sigkeys(prices_with_time[0].T.shape[1], 5))

# # Calculation of Expected signature component-wise
Exp_sign_iisignature = np.mean(sign_iisignature, axis=0)
Exp_sign_esig = np.mean(sign_esig, axis=0)

# # # # # # # # # # # # # # # # # # # # # # # #
# Calculations signatures of prices by signatory package
prices_with_time_tensor = torch.zeros((paths, steps, 2))
prices_with_time_tensor.shape

for i in range(paths):
    prices_with_time_tensor[i] = torch.tensor([prices[i], time_array]).T

sign_signatory = signatory.signature(prices_with_time_tensor, 3)
sign_signatory.shape

# Compare EXPECTED SIGNATURES
Exp_sign_iisignature = np.mean(sign_iisignature, axis=0)
Exp_sign_esig = np.mean(sign_esig, axis=0)
Exp_sign_signatory = torch.mean(sign_signatory, axis=0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculations of signatures with PRICES and SIGMAS vectors

prices_and_sigs_with_time_tensor = torch.zeros((paths, steps, 3))
# prices_and_sigs_with_time_tensor.shape

for i in range(paths):
    prices_and_sigs_with_time_tensor[i] = torch.tensor([prices[i], sigs[i], time_array]).T

# prices_and_sigs_with_time_tensor.shape
sign_ps_signatory = signatory.signature(prices_and_sigs_with_time_tensor, 2)
sign_ps_signatory.shape
Exp_sign_signatory = torch.mean(sign_ps_signatory, axis=0)

thetas = np.arange(0.01, 0.2, 0.05)
kappas = np.arange(3, 5)
v_0s = thetas  # = reversion_level
rhos = np.arange(0.1, 1.1, 0.3)
sigmas = np.arange(0.1, 1.1, 0.3)
rs = np.arange(0.01, 0.10, 0.03)

time_array = np.arange(steps)
prices_and_sigs_with_time_tensor = torch.zeros((paths, steps, 3))
number_of_loops = thetas.shape[0] * kappas.shape[0] * rhos.shape[0] * sigmas.shape[0] * rs.shape[0]

sign_ps_signatory_ranges = torch.zeros((number_of_loops, paths, 12))
Exp_sign_signatory_results = torch.zeros((number_of_loops, 12))
sign_ps_signatory_ranges[0].shape


def signatures_calculations(S, T, rs, kappas, thetas, v_0s, rhos, sigmas, steps, paths):
    count = 0
    for r in tqdm(rs):
        for rho in rhos:
            for sigma in sigmas:
                for kappa in kappas:
                    for theta in thetas:

                        prices_1, sigs_1 = generate_heston_paths(S, T, r, kappa, theta, theta, rho, sigma, steps, paths,
                                                                 return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        sign_ps_signatory_ranges[count] = signatory.signature(prices_and_sigs_with_time_tensor, 2)
                        Exp_sign_signatory_results[count] = torch.mean(sign_ps_signatory_ranges[count], axis=0)
                        count = count + 1
    return sign_ps_signatory_ranges, Exp_sign_signatory_results


results = signatures_calculations(S, T, rs, kappas, thetas, v_0s, rhos, sigmas, steps, paths)
signatures_results = results[0]
Exp_signatures_results = results[1]
Exp_signatures_results.shape

plt.figure(figsize=(8, 8))
plt.plot(Exp_signatures_results.T)
plt.show()
plt.figure(figsize=(8, 8))
plt.plot(signatures_results[0].T)
plt.show()
