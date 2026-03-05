import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple


class Model:
    """Model of correlation computing based on Time-Varying Parameter with stochastic volatility model.

    All parameters are passed explicitly to the constructor.
    """

    def __init__(self,
                 num_iters=2000,
                 burn_in=1800,
                 mu0=10000.0,
                 sigma_init=np.array([[0.015, 0.0], [0.0, 0.015]]),
                 sigma2_eta_init=0.05,
                 gamma_init=1.0,
                 phi_init=1.0,
                 sample_phi=False,
                 sample_gamma=False,
                 alpha_phi0=0.5,
                 beta_phi0=0.5,
                 nu0_sigma2_eta=2.0,
                 V0_sigma2_eta=1.0,
                 gamma_0=2.1,
                 V0_gamma=0.5,
                 seed=25):
        self.num_iters = num_iters
        self.burn_in = burn_in
        self.mu0 = mu0
        self.sigma_init = sigma_init.copy()
        self.sigma2_eta_init = sigma2_eta_init
        self.gamma_init = gamma_init
        self.phi_init = phi_init
        self.sample_phi = sample_phi
        self.sample_gamma = sample_gamma
        self.alpha_phi0 = alpha_phi0
        self.beta_phi0 = beta_phi0
        self.nu0_sigma2_eta = nu0_sigma2_eta
        self.V0_sigma2_eta = V0_sigma2_eta
        self.gamma_0 = gamma_0
        self.V0_gamma = V0_gamma
        self.seed = seed

    def run(self, z, x, progress_bar=True):
        """Run MCMC to estimate the TVP model.

        Parameters
        ----------
        z : np.ndarray of shape (n, 1)
            Dependent variable.
        x : np.ndarray of shape (n, 2)
            Design matrix (first column ones, second column factor).
        progress_bar : bool
            Whether to display a tqdm progress bar.

        Returns
        -------
        dict
            Dictionary containing MCMC samples.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        n = len(z)
        num_iters = self.num_iters

        # Initialize chains
        Sigma_est = self.sigma_init.copy()
        sigma2_eta_est = np.ones(num_iters) * self.sigma2_eta_init
        gamma_est = np.ones(num_iters) * self.gamma_init
        phi_est = np.ones(num_iters) * self.phi_init

        sigma_t_est = np.ones(n)

        m = np.zeros([num_iters, n, 2])
        sigma = np.zeros([num_iters, n, 2, 2])
        a_est = np.ones((num_iters, n, 2))

        h_est = np.zeros((num_iters, n))
        sigma_t_est_history = np.zeros((num_iters, n))

        iterator = tqdm(range(num_iters)) if progress_bar else range(num_iters)

        for k in iterator:
            # --- Sample a (coefficients) ---
            if k == 0:
                # Initialize first iteration
                for i in range(1, n):
                    p = 0
                    sigma[k, i, :, :] = np.linalg.inv(np.linalg.inv(Sigma_est) + np.dot((x[i, :].reshape(1, 2)).T, x[i, :].reshape(1, 2)) / sigma_t_est[i] ** 2)
                    m[k, i, :] = np.dot(np.dot((a_est[p, i - 1, :]).reshape(1, 2), np.linalg.inv(Sigma_est)) + z[i, 0] * x[i, :].reshape(1, 2) / sigma_t_est[i] ** 2, sigma[k, i, :, :])
                    a_est[k, i, :] = m[k, i, :]
            else:
                p = k - 1
                for i in range(n):
                    if (i > 0) and (i < (n - 1)):
                        sigma[k, i, :, :] = np.linalg.inv(2 * np.linalg.inv(Sigma_est) + np.dot((x[i, :].reshape(1, 2)).T, x[i, :].reshape(1, 2)) / sigma_t_est[i] ** 2)
                        m[k, i, :] = np.dot(np.dot((a_est[p, i - 1, :]).reshape(1, 2) + (a_est[p, i + 1, :]).reshape(1, 2), np.linalg.inv(Sigma_est)) + z[i, 0] * x[i, :].reshape(1, 2) / sigma_t_est[i] ** 2, sigma[k, i, :, :])
                        a_est[k, i, :] = np.random.multivariate_normal(m[k, i, :], sigma[k, i, :, :], 1)
                    elif i == 0:
                        sigma[k, i, :, :] = np.linalg.inv(np.linalg.inv(Sigma_est) + np.dot((x[i, :].reshape(1, 2)).T, x[i, :].reshape(1, 2)) / sigma_t_est[i] ** 2)
                        m[k, i, :] = np.dot(np.dot((a_est[p, i + 1, :]).reshape(1, 2), np.linalg.inv(Sigma_est)) + z[i, 0] * x[i, :].reshape(1, 2) / sigma_t_est[i] ** 2, sigma[k, i, :, :])
                        a_est[k, i, :] = np.random.multivariate_normal(m[k, i, :], sigma[k, i, :, :], 1)
                    elif i == n - 1:
                        sigma[k, i, :, :] = np.linalg.inv(np.linalg.inv(Sigma_est) + np.dot((x[i, :].reshape(1, 2)).T, x[i, :].reshape(1, 2)) / sigma_t_est[i] ** 2)
                        m[k, i, :] = np.dot(np.dot((a_est[p, i - 1, :]).reshape(1, 2), np.linalg.inv(Sigma_est)) + z[i, 0] * x[i, :].reshape(1, 2) / sigma_t_est[i] ** 2, sigma[k, i, :, :])
                        a_est[k, i, :] = np.random.multivariate_normal(m[k, i, :], sigma[k, i, :, :], 1)

            # --- Sample h (volatility latent states) ---
            for i in range(n):
                if (i > 0) and (i < (n - 1)):
                    h_est[k, i] = bs.get_single_mcmc_sample(
                        h_est[k - 1, i - 1], h_est[k - 1, i + 1],
                        phi_est[k - 1], sigma2_eta_est[k - 1],
                        z[i, 0], a_est[k, i, 0], a_est[k, i, 1],
                        x[i, 1], gamma_est[k - 1],
                        is_first=True, is_last=True, num_samples=1
                    )
                elif i == 0:
                    h_est[k, i] = bs.get_single_mcmc_sample(
                        0, h_est[k - 1, i + 1],
                        phi_est[k - 1], sigma2_eta_est[k - 1],
                        z[i, 0], a_est[k, i, 0], a_est[k, i, 1],
                        x[i, 1], gamma_est[k - 1],
                        is_first=False, is_last=True, num_samples=1
                    )
                elif i == n - 1:
                    h_est[k, i] = bs.get_single_mcmc_sample(
                        h_est[k - 1, i - 1], 0,
                        phi_est[k - 1], sigma2_eta_est[k - 1],
                        z[i, 0], a_est[k, i, 0], a_est[k, i, 1],
                        x[i, 1], gamma_est[k - 1],
                        is_first=True, is_last=False, num_samples=1
                    )

            # --- Sample Sigma ---
            Sigma_est = bs.sample_Sigma(a_est[k, :, :], Sigma_est, self.mu0)

            # --- Sample phi (optional) ---
            if self.sample_phi:
                phi_est[k] = bs.sample_phi_mh(
                    phi_est[k - 1], h_est[k, :], sigma2_eta_est[k - 1],
                    self.alpha_phi0, self.beta_phi0
                )
            elif k > 0:
                phi_est[k] = phi_est[k - 1]

            # --- Sample sigma2_eta ---
            sigma2_eta_est[k] = bs.sample_sigma2_eta_gibbs(
                h_est[k, :], phi_est[k],
                self.nu0_sigma2_eta, self.V0_sigma2_eta
            )

            # --- Sample gamma (optional) ---
            if self.sample_gamma:
                gamma_est[k] = bs.sample_gamma_gibbs(
                    z, x, a_est[k, :, :], h_est[k, :],
                    self.gamma_0, self.V0_gamma
                )
            else:
                gamma_est[k] = gamma_est[k - 1]

            # --- Update conditional variance sigma_t^2 ---
            sigma_t_est = (gamma_est[k] * np.exp(h_est[k, :])) ** 0.5
            sigma_t_est_history[k, :] = sigma_t_est ** 2

            if progress_bar and k % 500 == 0 and k > 0:
                iterator.set_description(
                    f"Iter {k}, sigma2_eta: {sigma2_eta_est[k]:.4f}, "
                    f"phi: {phi_est[k]:.4f}, gamma: {gamma_est[k]:.4f}"
                )

        results = {
            'a_est': a_est,
            'h_est': h_est,
            'sigma_t_est_history': sigma_t_est_history,
            'sigma2_eta_est': sigma2_eta_est,
            'gamma_est': gamma_est,
            'phi_est': phi_est,
            'sigma_est': Sigma_est
        }
        return results

    def compute_correlation(self, results, burn_in=None, x_variance=None):
        """Compute dynamic correlation from MCMC results.

        Parameters
        ----------
        results : dict
            Dictionary returned by the `run` method.
        burn_in : int, optional
            Number of burn-in iterations. If None, uses self.burn_in.
        x_variance : float or array, optional
            Variance of the factor x. If None, assumes constant 1.

        Returns
        -------
        r_mean : np.ndarray
            Posterior mean correlation.
        r_low : np.ndarray
            Lower 2.5% credible interval.
        r_up : np.ndarray
            Upper 97.5% credible interval.
        """
        if burn_in is None:
            burn_in = self.burn_in

        a_est = results['a_est']
        sigma_t_est = results['sigma_t_est_history']

        a_post = a_est[burn_in:, :, 1]          # a1 coefficients
        sigma_t_post = sigma_t_est[burn_in:, :]

        n_samples, n_time = a_post.shape

        if x_variance is None:
            x_var = np.ones(n_time)
        elif np.isscalar(x_variance):
            x_var = np.ones(n_time) * x_variance
        else:
            x_var = np.asarray(x_variance)
            if len(x_var) != n_time:
                print(f"Warning: x_variance length ({len(x_var)}) != n_time ({n_time}), interpolating.")
                x_var = np.interp(
                    np.linspace(0, n_time - 1, n_time),
                    np.linspace(0, len(x_var) - 1, len(x_var)),
                    x_var,
                )

        x_var_reshaped = x_var.reshape(1, -1)

        numerator = a_post * np.sqrt(x_var_reshaped)
        denominator = np.sqrt(a_post ** 2 * x_var_reshaped + sigma_t_post)
        corr_samples = numerator / denominator
        corr_samples = np.clip(corr_samples, -1.0, 1.0)

        r_mean = np.mean(corr_samples, axis=0)
        r_low = np.percentile(corr_samples, 2.5, axis=0)
        r_up = np.percentile(corr_samples, 97.5, axis=0)

        return r_mean, r_low, r_up
