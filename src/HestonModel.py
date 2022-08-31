import numpy as np
from tqdm.auto import tqdm


def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, sigma, steps, Npaths, return_vol=False):
    size = (Npaths, steps)
    sigs = np.zeros(size)
    prices = np.zeros(size)
    dt = T / steps
    S_t = S
    v_t = v_0
    for t in tqdm(range(steps)):
        # simulate stochastic processes
        WT = np.random.multivariate_normal(np.array([0, 0]), cov=np.array([[1, rho], [rho, 1]]), size=Npaths) * np.sqrt(
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


