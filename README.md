# TVP Correlation

A lightweight Python package for estimating time‑varying correlation using a Bayesian Time‑Varying Parameter (TVP) model with stochastic volatility.
The model is defined as:

$$
\begin{aligned}
z_t = a_{0t} + a_{1t} x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_t^2), \\
a_t = a_{t-1} + u_t, \quad u_t \sim \mathcal{N}(0, \Sigma), \\
\sigma_t^2 = \gamma \exp(h_t), \\
h_{t+1} = \phi h_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2),
\end{aligned}
$$

and the time‑varying correlation is computed as

$$
r_t = \frac{a_{1t} \cdot \sigma_{x,t}}{\sqrt{a_{1t}^2 \cdot \sigma_{x,t}^2 + \sigma_t^2}}.
$$

The package provides a flexible MCMC sampler (Metropolis‑Hastings + Gibbs) to estimate the latent states and parameters. All model hyperparameters are passed explicitly – no hidden configuration files.

## Features
- Bayesian estimation of TVP coefficients and stochastic volatility.

- Computation of dynamic correlation with credible intervals.

## Installation

### From GitHub

```bash
pip install git+https://github.com/yourusername/tvp-correlation.git
```



## Quick Start

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
```
## API Reference

#### TVPModel

The main class for the TVP model.


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iters` | int | 2000 | Number of MCMC iterations. |
| `burn_in` | int | 1800 | Number of burn‑in iterations (used in `compute_correlation`). |
| `mu0` | float | 10000.0 | Prior degrees of freedom for the covariance matrix $\Sigma$ (inverse Wishart). |
| `sigma_init` | (2,2) array | `[[0.015,0],[0,0.015]]` | Initial value for $\Sigma$. |
| `sigma2_eta_init` | float | 0.05 | Initial value for $\sigma_\eta^2$ (innovation variance of log‑volatility). |
| `gamma_init` | float | 1.0 | Initial value for $\gamma$ (scale of volatility). |
| `phi_init` | float | 1.0 | Initial value for $\phi$ (AR coefficient of log‑volatility). |
| `sample_phi` | bool | False | Whether to sample $\phi$ (if False, $\phi$ is fixed at its initial value). |
| `sample_gamma` | bool | False | Whether to sample $\gamma$ (if False, $\gamma$ is fixed). |
| `alpha_phi0` | float | 0.5 | First shape parameter of the Beta prior on $(\phi+1)/2$. |
| `beta_phi0` | float | 0.5 | Second shape parameter of the Beta prior. |
| `nu0_sigma2_eta` | float | 2.0 | Prior degrees of freedom for $\sigma_\eta^2$ (inverse gamma). |
| `V0_sigma2_eta` | float | 1.0 | Prior scale for $\sigma_\eta^2$ (inverse gamma). |
| `gamma_0` | float | 2.1 | Prior shape for $\gamma$ (inverse gamma). |
| `V0_gamma` | float | 0.5 | Prior scale for $\gamma$ (inverse gamma). |
| `seed` | int or None | 25 | Random seed for reproducibility. |

## Methods

#### `run(z, x, progress_bar=True)`
Runs the MCMC algorithm.

**Parameters:**
- **z** : array of shape `(n, 1)`  
  Dependent variable.
- **x** : array of shape `(n, 2)`  
  Design matrix (first column ones, second column the factor).
- **progress_bar** : bool, default=`True`  
  Whether to show a tqdm progress bar.

**Returns:**  
A dictionary containing all MCMC samples with the following keys:
- `'a_est'` — samples of time‑varying coefficients, shape `(num_iters, n, 2)`
- `'h_est'` — samples of log‑volatility, shape `(num_iters, n)`
- `'sigma_t_est_history'` — samples of conditional variance $\sigma_t^2$, shape `(num_iters, n)`
- `'sigma2_eta_est'` — samples of $\sigma_\eta^2$, shape `(num_iters,)`
- `'gamma_est'` — samples of $\gamma$, shape `(num_iters,)`
- `'phi_est'` — samples of $\phi$, shape `(num_iters,)`
- `'sigma_est'` — final estimate of the covariance matrix $\Sigma$ (last iteration)

---

#### `compute_correlation(results, burn_in=None, x_variance=None)`
Computes the dynamic correlation from the MCMC samples.

**Parameters:**
- **results** : dict  
  The dictionary returned by `run()`.
- **burn_in** : int, optional  
  Number of burn‑in iterations to discard. If `None`, uses the value from the constructor.
- **x_variance** : float or array, optional  
  Variance of the factor $x$. If a float, assumed constant over time. If an array, must have length `n`. If `None`, assumes constant 1.

**Returns:**  
A tuple `(r_mean, r_low, r_up)` where each element is a NumPy array of length `n`:
- `r_mean` — posterior mean correlation
- `r_low` — lower bound of the 95% credible interval (2.5th percentile)
- `r_up` — upper bound of the 95% credible interval (97.5th percentile)
