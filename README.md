TVP Correlation
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/python-3.9+-blue.svg

A lightweight Python package for estimating time‑varying correlation using a Bayesian Time‑Varying Parameter (TVP) model with stochastic volatility.
The model is defined as:

$$
\begin{aligned}
z_t = a_{0t} + a_{1t} x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_t^2), \\
a_t = a_{t-1} + u_t, \quad u_t \sim \mathcal{N}(0, \Sigma), \\
\sigma_t^2 &= \gamma \exp(h_t), \\
h_{t+1} = \phi h_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2),
\end{aligned}
$$

$$
r_t = \frac{a_{1t} \cdot \sigma_{x,t}}{\sqrt{a_{1t}^2 \cdot \sigma_{x,t}^2 + \sigma_t^2}}.
$$

The package provides a flexible MCMC sampler (Metropolis‑Hastings + Gibbs) to estimate the latent states and parameters. All model hyperparameters are passed explicitly – no hidden configuration files.

Features
Bayesian estimation of TVP coefficients and stochastic volatility.

Computation of dynamic correlation with credible intervals.

Installation
From GitHub
bash
pip install git+https://github.com/yourusername/tvp-correlation.git
Editable install for development
bash
git clone https://github.com/yourusername/tvp-correlation.git
cd tvp-correlation
pip install -e .

Quick Start
```python
import numpy as np
from tvp_correlation import TVPModel

# Generate some synthetic data (or load your own)
n = 500
x = np.random.randn(n, 1)                     # factor variable
z = 0.5 + 0.3 * x + 0.2 * np.random.randn(n, 1)  # dependent variable (simplified)

# Design matrix: first column ones (intercept), second column the factor
x_design = np.hstack([np.ones((n, 1)), x])

# Create model with custom settings
model = TVPModel(
    num_iters=2000,          # number of MCMC iterations
    burn_in=1800,            # burn-in period
    sample_phi=False,        # fix phi to initial value (no sampling)
    sample_gamma=False,      # fix gamma to initial value
    seed=42                  # for reproducibility
)

# Run MCMC
results = model.run(z, x_design, progress_bar=True)

# Compute dynamic correlation (assuming variance of x = 1)
r_mean, r_low, r_up = model.compute_correlation(results)

# r_mean, r_low, r_up are arrays of length n
# You can plot them or use them in further analysis
API Reference
TVPModel
The main class for the TVP model.


Parameter	Type	Default	Description
num_iters	int	2000	Number of MCMC iterations.
burn_in	int	1800	Number of burn‑in iterations (used in compute_correlation).
mu0	float	10000.0	Prior degrees of freedom for the covariance matrix $\Sigma$ (inverse Wishart).
sigma_init	(2,2) array	[[0.015,0],[0,0.015]]	Initial value for $\Sigma$.
sigma2_eta_init	float	0.05	Initial value for $\sigma_\eta^2$ (innovation variance of log‑volatility).
gamma_init	float	1.0	Initial value for $\gamma$ (scale of volatility).
phi_init	float	1.0	Initial value for $\phi$ (AR coefficient of log‑volatility).
sample_phi	bool	False	Whether to sample $\phi$ (if False, $\phi$ is fixed at its initial value).
sample_gamma	bool	False	Whether to sample $\gamma$ (if False, $\gamma$ is fixed).
alpha_phi0	float	0.5	First shape parameter of the Beta prior on $(\phi+1)/2$.
beta_phi0	float	0.5	Second shape parameter of the Beta prior.
nu0_sigma2_eta	float	2.0	Prior degrees of freedom for $\sigma_\eta^2$ (inverse gamma).
V0_sigma2_eta	float	1.0	Prior scale for $\sigma_\eta^2$ (inverse gamma).
gamma_0	float	2.1	Prior shape for $\gamma$ (inverse gamma).
V0_gamma	float	0.5	Prior scale for $\gamma$ (inverse gamma).
seed	int or None	25	Random seed for reproducibility.
Methods
run(z, x, progress_bar=True)
Runs the MCMC algorithm.

z: array of shape (n, 1) – dependent variable.

x: array of shape (n, 2) – design matrix (first column ones, second column the factor).

progress_bar: bool – show tqdm progress bar.
Returns a dictionary containing all MCMC samples (keys: 'a_est', 'h_est', 'sigma_t_est_history', 'sigma2_eta_est', 'gamma_est', 'phi_est', 'sigma_est').

compute_correlation(results, burn_in=None, x_variance=None)
Computes the dynamic correlation from the MCMC samples.

results: the dictionary returned by run().

burn_in: int – if None, uses the value from the constructor.

x_variance: float or array – variance of the factor x. If None, assumes constant 1.
Returns (r_mean, r_low, r_up) where each is an array of length n.

License
This project is licensed under the MIT License – see the LICENSE file for details.

Citation
If you use this code in your research, please cite:

bibtex
@software{tvp_correlation,
  author = {Your Name},
  title = {TVP Correlation: A Python Package for Time-Varying Correlation},
  year = {2025},
  url = {https://github.com/yourusername/tvp-correlation}
}
