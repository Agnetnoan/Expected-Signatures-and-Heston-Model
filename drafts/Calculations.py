from src.HestonModel import generate_heston_paths
import numpy as np
import esig
import matplotlib.pyplot as plt
import iisignature
import signatory
import torch
from tqdm.auto import tqdm


# Case 0: Heston model with constant parameters (esig, iisignatures, signatory)
kappa = 3
theta = 0.02
v_0 = theta #
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

for i in range(paths):
    prices_with_time[i] = np.array([prices[i], time_array])

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
prices_with_time_tensor = torch.zeros((paths,steps, 2 ))
prices_with_time_tensor.shape

for i in range(paths):
    prices_with_time_tensor[i] = torch.tensor([prices[i], time_array]).T

sign_signatory=signatory.signature(prices_with_time_tensor,3)
sign_signatory.shape

# Compare EXPECTED SIGNATURES
Exp_sign_iisignature = np.mean(sign_iisignature, axis=0)
Exp_sign_esig = np.mean(sign_esig, axis=0)
Exp_sign_signatory = torch.mean(sign_signatory, axis=0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculations of signatures with PRICES and SIGMAS vectors

prices_and_sigs_with_time_tensor = torch.zeros((paths,steps, 3 ))
# prices_and_sigs_with_time_tensor.shape

for i in range(paths):
    prices_and_sigs_with_time_tensor[i] = torch.tensor([prices[i],sigs[i], time_array]).T

# prices_and_sigs_with_time_tensor.shape
sign_ps_signatory=signatory.signature(prices_and_sigs_with_time_tensor,2)
sign_ps_signatory.shape
Exp_sign_signatory = torch.mean(sign_ps_signatory, axis=0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Case 1: time_array+prices+sigmas
thetas=np.arange(0.01,0.2,0.05)
kappas = np.arange(3,5)
v_0s = thetas #  = reversion_level
rhos = np.arange(0.1,1.1,0.3)
sigmas = np.arange(0.1,1.1,0.3)
rs = np.arange(0.01,0.10,0.03)

depth_of_sig=2
width_of_sig=3
sig_keys = esig.sigkeys(width_of_sig, depth_of_sig)
sig_dim=signatory.signature_channels(width_of_sig,depth_of_sig)
# define time_array and empty tensor for prices and sigmas, signatures and expected signatures
time_array = np.arange(steps)
prices_and_sigs_with_time_tensor = torch.zeros((paths,steps, width_of_sig ))
number_of_loops=thetas.shape[0]*kappas.shape[0]*rhos.shape[0]*sigmas.shape[0]*rs.shape[0]
sign_ps_signatory_differentranges=torch.zeros((number_of_loops,paths,sig_dim))
Exp_sign_signatory_results=torch.zeros((number_of_loops,sig_dim))
sign_ps_signatory_differentranges[0].shape

def signatures_calculations_with_timearray(S, T, rs, kappas, thetas, v_0s, rhos, sigmas, steps, paths):
    count=0
    for r in tqdm(rs):
        for rho in rhos:
            for sigma in sigmas:
                for kappa in kappas:
                    for theta in thetas:
                        prices_1, sigs_1 = generate_heston_paths(S, T, r, kappa, theta, theta, rho, sigma, steps, paths,
                                                                 return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        sign_ps_signatory_differentranges[count] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        Exp_sign_signatory_results[count] = torch.mean(sign_ps_signatory_differentranges[count], axis=0)
                        count = count + 1
    return sign_ps_signatory_differentranges,Exp_sign_signatory_results

results_3=signatures_calculations_with_timearray(S, T, rs, kappas, thetas, v_0s, rhos, sigmas, steps, paths)
signatures_results_3=results_3[0]
Exp_signatures_results_3=results_3[1]
# Exp_signatures_results_3.shape
# signatures_results_3.shape

plt.figure(figsize=(8, 8))
plt.plot(Exp_signatures_results_3.T)
plt.title("Expeceted signatures with time array")
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(signatures_results_3[0].T)
plt.title("Signatures with initial parameters with time array")
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(Exp_signatures_results_3.T[0])
plt.title("Parameter [0] of all expected signature with time array")
plt.show()

# ####################################################################
# Case 2: without time_array. only prices+sigmas
S = 100
paths = 10000
steps = 100
T = 1

thetas=np.arange(0.01,0.2,0.05)
kappas = np.arange(3,5)
v_0s = thetas #  = reversion_level
rhos = np.arange(0.1,1.1,0.3)
sigmas = np.arange(0.1,1.1,0.3)
rs = np.arange(0.01,0.10,0.03)

depth_of_sig=3  # depth of the singature
width_of_sig=2  # width of the singature(prices+sigmas)
sig_keys = esig.sigkeys(width_of_sig, depth_of_sig)
sig_dim=signatory.signature_channels(width_of_sig,depth_of_sig)

# define time_array and empty tensor for prices and sigmas, signatures and expected signatures
time_array = np.arange(steps)
prices_and_sigs_without_time_tensor = torch.zeros((paths,steps, width_of_sig ))
number_of_loops=thetas.shape[0]*kappas.shape[0]*rhos.shape[0]*sigmas.shape[0]*rs.shape[0]
sign_ps_signatory_differentranges=torch.zeros((number_of_loops,paths,sig_dim))
Exp_sign_signatory_results=torch.zeros((number_of_loops,sig_dim))
sign_ps_signatory_differentranges[0].shape


def signatures_calculations(S, T, rs, kappas, thetas, v_0s, rhos, sigmas, steps, paths):
    count=0
    for r in tqdm(rs):
        for rho in rhos:
            for sigma in sigmas:
                for kappa in kappas:
                    for theta in thetas:
                        prices_1, sigs_1 = generate_heston_paths(S, T, r, kappa, theta, theta, rho, sigma, steps, paths,
                                                                 return_vol=True)
                        for j in range(paths):
                             prices_and_sigs_without_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j]]).T
                        #     shape of prices_and_sigs_without_time_tensor=torch.Size([10000, 100, 2])
                        sign_ps_signatory_differentranges[count] = signatory.signature(prices_and_sigs_without_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([512, 10000, 14])
                        Exp_sign_signatory_results[count] = torch.mean(sign_ps_signatory_differentranges[count], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([512, 14])
                        count = count + 1
    return sign_ps_signatory_differentranges,Exp_sign_signatory_results

results=signatures_calculations(S, T, rs, kappas, thetas, v_0s, rhos, sigmas, steps, paths)
signatures_results=results[0]
Exp_signatures_results=results[1]
# signatures_results.shape
# signatures_results[0].shape
# Exp_signatures_results.shape

# Plots
plt.figure(figsize=(8, 8))
plt.plot(Exp_signatures_results.T)
plt.title("Expected signatures")
plt.show()
plt.figure(figsize=(8, 8))
plt.plot(signatures_results[0].T)
plt.title("Signatures with initial parameters")
plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(signatures_results[100].T)
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(signatures_results[200].T)

plt.figure(figsize=(8, 8))
plt.plot(Exp_signatures_results.T[0])
plt.title("Parameter [0] of all expected signatures")
plt.show()
plt.figure(figsize=(8, 8))
plt.plot(Exp_signatures_results.T[1])
plt.title("Parameter [1] of all expected signatures")
plt.show()
plt.figure(figsize=(8, 8))
plt.plot(Exp_signatures_results.T[2])
plt.title("Parameter [2] of all expected signatures")
plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[3])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[4])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[5])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[6])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[7])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[8])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[9])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[10])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[11])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[12])
# plt.show()
# plt.figure(figsize=(8, 8))
# plt.plot(Exp_signatures_results.T[13])
# plt.show()






