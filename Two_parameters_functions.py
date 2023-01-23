from src.HestonModel import generate_heston_paths
import numpy as np
import esig
import matplotlib.pyplot as plt
import signatory
import torch

# constant Heston parameters
kappa = 2
theta = 0.15
v_0 = theta #
rho = 0.6
sigma = 0.6
r = 0.04

S = 100
paths = 10000
steps = 50
T = 1

depth_of_sig=2  #level of truncation of signatures
width_of_sig=3  # prices + sigmas + time_array (parameters for signatures)
sig_keys = esig.sigkeys(width_of_sig, depth_of_sig)
sig_dim=signatory.signature_channels(width_of_sig,depth_of_sig)

by_parameter_2 = {
    "by_rs": np.linspace(0.01,0.15,50),  # list to store 50 elements for var1
    "by_rhos": np.linspace(0.0,1.0,50),  # list to store 50 elements for var2
    "by_sigmas": np.linspace(0.0,1.1,50),  # list to store 50 elements for var3
    "by_kappas": np.linspace(.5,5.0,50),   # list to store 50 elements for var4
    "by_thetas": np.linspace(0.01,0.8,50)  # list to store 50 elements for var5
}
# Define time_array and empty tensor for prices and sigmas, signatures and expected signatures
time_array = np.arange(steps)
prices_and_sigs_with_time_tensor = torch.zeros((paths,steps, width_of_sig ))


def signatures_calculations_with_2parameters(S, T, rs, kappas, thetas, v_0s, rhos, sigmas, steps, paths, par1, par2):
    m=0

    if par1=='kappa' and par2=='theta':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_kappas"].shape[0],by_parameter_2["by_thetas"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_kappas"].shape[0],by_parameter_2["by_thetas"].shape[0],sig_dim))
        for kappa in by_parameter_2["by_kappas"]:
              n=0
              for theta in by_parameter_2["by_thetas"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappa, theta, theta, rhos, sigmas, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 50, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='theta' and par2=='kappa':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_thetas"].shape[0],by_parameter_2["by_kappas"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_thetas"].shape[0],by_parameter_2["by_kappas"].shape[0],sig_dim))
        for theta in by_parameter_2["by_thetas"]:
              n=0
              for kappa in by_parameter_2["by_kappas"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappa, theta, theta, rhos, sigmas, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='theta' and par2=='sigma':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_thetas"].shape[0],by_parameter_2["by_sigmas"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_thetas"].shape[0],by_parameter_2["by_sigmas"].shape[0],sig_dim))
        for theta in by_parameter_2["by_thetas"]:
              n=0
              for sigma in by_parameter_2["by_sigmas"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappas, theta, theta, rhos, sigma, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='sigma' and par2=='theta':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_sigmas"].shape[0],by_parameter_2["by_thetas"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_sigmas"].shape[0],by_parameter_2["by_thetas"].shape[0],sig_dim))
        for sigma in by_parameter_2["by_sigmas"]:
              n=0
              for theta in by_parameter_2["by_thetas"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappas, theta, theta, rhos, sigma, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='theta' and par2=='rho':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_thetas"].shape[0],by_parameter_2["by_rhos"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_thetas"].shape[0],by_parameter_2["by_rhos"].shape[0],sig_dim))
        for theta in by_parameter_2["by_thetas"]:
              n=0
              for rho in by_parameter_2["by_rhos"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappas, theta, theta, rho, sigmas, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='rho' and par2=='theta':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_rhos"].shape[0],by_parameter_2["by_thetas"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_rhos"].shape[0],by_parameter_2["by_thetas"].shape[0],sig_dim))
        for rho in by_parameter_2["by_rhos"]:
              n=0
              for theta in by_parameter_2["by_thetas"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappas, theta, theta, rho, sigmas, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1

    if par1=='sigma' and par2=='kappa':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_sigmas"].shape[0],by_parameter_2["by_kappas"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_sigmas"].shape[0],by_parameter_2["by_kappas"].shape[0],sig_dim))
        for sigma in by_parameter_2["by_sigmas"]:
              n=0
              for kappa in by_parameter_2["by_kappas"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappa, thetas, thetas, rhos, sigma, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='kappa' and par2=='sigma':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_kappas"].shape[0],by_parameter_2["by_sigmas"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_kappas"].shape[0],by_parameter_2["by_sigmas"].shape[0],sig_dim))
        for kappa in by_parameter_2["by_kappas"]:
              n=0
              for sigma in by_parameter_2["by_sigmas"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappa, thetas, thetas, rhos, sigma, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='kappa' and par2=='rho':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_kappas"].shape[0],by_parameter_2["by_rhos"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_kappas"].shape[0],by_parameter_2["by_rhos"].shape[0],sig_dim))
        for kappa in by_parameter_2["by_kappas"]:
              n=0
              for rho in by_parameter_2["by_rhos"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappa, thetas, thetas, rho, sigmas, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='rho' and par2=='kappa':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_rhos"].shape[0],by_parameter_2["by_kappas"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_rhos"].shape[0],by_parameter_2["by_kappas"].shape[0],sig_dim))
        for rho in by_parameter_2["by_rhos"]:
              n=0
              for kappa in by_parameter_2["by_kappas"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappa, thetas, thetas, rho, sigmas, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='rho' and par2=='sigma':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_rhos"].shape[0],by_parameter_2["by_sigmas"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_rhos"].shape[0],by_parameter_2["by_sigmas"].shape[0],sig_dim))
        for rho in by_parameter_2["by_rhos"]:
              n=0
              for sigma in by_parameter_2["by_sigmas"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappas, thetas, thetas, rho, sigma, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    if par1=='sigma' and par2=='rho':
        sign_signatory_2_parameters=torch.zeros((by_parameter_2["by_sigmas"].shape[0],by_parameter_2["by_rhos"].shape[0],paths,sig_dim))
        Exp_sign_signatory_results_2_parameters=torch.zeros((by_parameter_2["by_sigmas"].shape[0],by_parameter_2["by_rhos"].shape[0],sig_dim))
        for sigma in by_parameter_2["by_sigmas"]:
              n=0
              for rho in by_parameter_2["by_rhos"]:
                        # Caclulation of 10000 paths of prices and sigmas
                        prices_1, sigs_1 = generate_heston_paths(S, T, rs, kappas, thetas, thetas, rho, sigma, steps, paths, return_vol=True)
                        for j in range(paths):
                            prices_and_sigs_with_time_tensor[j] = torch.tensor([prices_1[j], sigs_1[j], time_array]).T
                        #     shape of prices_and_sigs_witout_time_tensor=torch.Size([10000, 100, 3])

                        sign_signatory_2_parameters[m][n] = signatory.signature(prices_and_sigs_with_time_tensor, depth_of_sig)
                        #     shape of  sign_ps_signatory_differentranges=torch.Size([50, 50, 10000, 12])

                        Exp_sign_signatory_results_2_parameters[m][n] = torch.mean(sign_signatory_2_parameters[m][n], axis=0)
                        #     shape of  Exp_sign_signatory_results=torch.Size([50, 50, 12])

                        n=n+1
              m=m+1
    return sign_signatory_2_parameters,Exp_sign_signatory_results_2_parameters

