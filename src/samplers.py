import numpy as np
from scipy.stats import norm, t, wishart, invgamma, truncnorm
from scipy.optimize import minimize_scalar
from scipy.special import betaln
import scipy.stats as stats


def get_single_mcmc_sample(ht_1, ht_2, phi, sigma2_eta_2, zt, alpha0, alpha1, xt, gamma,
                           is_first=True, is_last=True, num_samples=1):
    """
    Independent Metropolis-Hastings Sampler for a single h_t.
    Reference: Hastings (1970) or any MCMC textbook
    """
    def log_target(h):
        log_lik = -0.5 * h - 0.5 * (zt - alpha0 - alpha1 * xt)**2 / (gamma * np.exp(h))

        # Log-prior from AR(1) process
        log_prior = 0.0

        if is_first:
            log_prior += -0.5 * (h - ht_1 * phi)**2 / sigma2_eta_2

        if is_last:
            log_prior += -0.5 * (ht_2 - phi * h)**2 / sigma2_eta_2

        return log_lik + log_prior

    # Find optimum for proposal construction
    result = minimize_scalar(lambda h: -log_target(h),
                           bounds=(min(ht_1*phi, ht_2/phi)-3,
                                  max(ht_1*phi, ht_2/phi)+3),
                           method='bounded')
    mode = result.x

    # Fixed proposal distribution (can be tuned)
    proposal_std = 1.0

    samples = []
    current = mode

    for _ in range(num_samples):
        # Independent proposal
        candidate = np.random.normal(mode, proposal_std)

        # Metropolis-Hastings ratio
        log_alpha = (log_target(candidate)
                     - log_target(current)
                     - 0.5*((candidate - mode)/proposal_std)**2
                     + 0.5*((current - mode)/proposal_std)**2)

        if np.log(np.random.uniform()) < log_alpha:
            current = candidate

        samples.append(current)

    return samples[0] if num_samples == 1 else samples


def sample_Sigma(a, Sigma_est, mu0):
    """Sample covariance matrix Sigma from posterior inverse Wishart."""
    n = len(a)
    df = mu0 + n - 1

    Omega = np.zeros((2, 2))
    for h in range(1, n):
        diff = (a[h, :] - a[h-1, :]).reshape(2, 1)
        Omega += diff @ diff.T

    theta = Sigma_est + Omega
    sigma_new = np.linalg.inv(wishart.rvs(df, scale=np.linalg.inv(theta)))

    return sigma_new


def sample_h_single(ht_1, ht_2, phi, sigma2_eta_2, zt, alpha0, alpha1, xt, gamma):
    """Sample a single h value using Independent Metropolis-Hastings."""
    def log_target(h):
        return (-0.5 * (h - ht_1 * phi)**2 / sigma2_eta_2
                - 0.5 * (h - (ht_2 / phi))**2 / (sigma2_eta_2 / phi**2)
                - 0.5 * h
                - 0.5 * (zt - alpha0 - alpha1 * xt)**2 / (gamma * np.exp(h)))

    # Find optimum for proposal construction
    result = minimize_scalar(
        lambda h: -log_target(h),
        bounds=(min(ht_1*phi, ht_2/phi)-3, max(ht_1*phi, ht_2/phi)+3),
        method='bounded'
    )
    mode = result.x

    # Fixed proposal distribution
    proposal_std = 1.0

    # Independent Metropolis-Hastings
    current = mode
    for _ in range(100):  # Multiple steps for better mixing
        candidate = np.random.normal(mode, proposal_std)

        log_alpha = (log_target(candidate)
                     - log_target(current)
                     - 0.5*((candidate - mode)/proposal_std)**2
                     + 0.5*((current - mode)/proposal_std)**2)

        if np.log(np.random.uniform()) < log_alpha:
            current = candidate

    return current


def sample_phi_mh(phi_current, h, sigma2_eta, alpha_phi0, beta_phi0, n_iter=100):
    """Sample phi using Metropolis-Hastings."""
    n = len(h)

    # Calculate proposal distribution parameters
    sum_htht1 = np.sum(h[:-1] * h[1:])
    sum_ht2 = np.sum(h[1:-1]**2) if n > 2 else 1.0

    mu_phi = sum_htht1 / sum_ht2 if sum_ht2 > 0 else 0.0
    sigma_phi = np.sqrt(sigma2_eta / sum_ht2) if sum_ht2 > 0 else 1.0

    def log_prior_phi(phi):
        """Log prior distribution for phi (Beta prior on (phi+1)/2)."""
        if np.abs(phi) >= 1:
            return -np.inf
        x = (phi + 1) / 2
        log_prior = (alpha_phi0-1)*np.log(x) + (beta_phi0-1)*np.log(1-x) - betaln(alpha_phi0, beta_phi0)
        return log_prior

    phi_samples = []
    phi_cur = phi_current

    for _ in range(n_iter):
        a_trunc = (-1 - mu_phi) / sigma_phi
        b_trunc = (1 - mu_phi) / sigma_phi
        phi_candidate = truncnorm.rvs(a_trunc, b_trunc, loc=mu_phi, scale=sigma_phi)

        log_prior_candidate = log_prior_phi(phi_candidate)
        log_prior_current = log_prior_phi(phi_cur)

        if np.isinf(log_prior_candidate) or np.isinf(log_prior_current):
            accept_prob = 0
        else:
            log_accept_ratio = (log_prior_candidate + 0.5*np.log(1-phi_candidate**2)) - \
                               (log_prior_current + 0.5*np.log(1-phi_cur**2))
            accept_prob = min(1, np.exp(log_accept_ratio))

        if np.random.rand() < accept_prob:
            phi_cur = phi_candidate

        phi_samples.append(phi_cur)

    return np.mean(phi_samples[-10:])


def sample_gamma_gibbs(z, x, a, h, gamma_0, V0):
    """Sample gamma using Gibbs sampling (inverse gamma)."""
    n = len(z)
    nu = gamma_0 + n
    V = V0

    for i in range(n):
        residual = (z[i, 0] - x[i, 1]*a[i, 1] - a[i, 0]) ** 2
        V += residual / np.exp(h[i])

    gamma_new = invgamma.rvs(a=nu/2, scale=V/2)
    return gamma_new


def sample_sigma2_eta_gibbs(h, phi, nu_0, V0):
    """Sample sigma2_eta using Gibbs sampling (inverse gamma)."""
    n = len(h)
    nu_hat = nu_0 + n
    V_hat = V0

    for t in range(n - 1):
        V_hat += (h[t + 1] - phi * h[t]) ** 2

    sigma2_eta_new = invgamma.rvs(a=nu_hat/2, scale=V_hat/2)
    return sigma2_eta_new
