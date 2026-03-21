"""
Probabilistic distribution heads for conflict fatality prediction.

Zero-Inflated Log-Normal Mixture (ZILNM) handles:
- 92% zeros (most country-months are peaceful)
- Extreme right skew (skewness=17, kurtosis=7810)
- Heavy tails (mass atrocity events with 10K+ fatalities)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInflatedLogNormalMixture(nn.Module):
    """
    Zero-Inflated Log-Normal Mixture distribution head.

    Outputs:
    - pi_zero: P(zero fatalities)
    - For nonzero: mixture of K log-normals for different violence regimes
    """

    def __init__(self, hidden_dim: int, n_components: int = 3):
        super().__init__()
        self.n_components = n_components

        # P(zero fatalities)
        self.zero_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Mixture weights + log-normal parameters
        self.mix_head = nn.Linear(hidden_dim, n_components)
        self.mu_head = nn.Linear(hidden_dim, n_components)
        self.log_sigma_head = nn.Linear(hidden_dim, n_components)

        # Initialize to match data priors
        with torch.no_grad():
            self.zero_head[-1].bias.fill_(2.3)  # sigmoid(2.3) ≈ 0.91
            self.mu_head.bias.copy_(torch.linspace(1, 7, n_components))
            self.log_sigma_head.bias.copy_(torch.zeros(n_components))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        pi_zero = torch.sigmoid(self.zero_head(h))
        mix_weights = F.softmax(self.mix_head(h), dim=-1)
        mus = self.mu_head(h)
        sigmas = 0.1 + 4.9 * torch.sigmoid(self.log_sigma_head(h))

        return {
            "pi_zero": pi_zero,
            "mix_weights": mix_weights,
            "mus": mus,
            "sigmas": sigmas,
        }

    def log_prob(self, params: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """Log-probability of observations under the ZILNM distribution."""
        pi_zero = params["pi_zero"].squeeze(-1)
        mix_weights = params["mix_weights"]
        mus = params["mus"]
        sigmas = params["sigmas"]

        zero_mask = (y == 0).float()

        log_p_zero = torch.log(pi_zero + 1e-8)

        # Positive case: mixture of log-normals
        log_y = torch.log(y.clamp(min=1e-6))
        log_y_expanded = log_y.unsqueeze(-1)
        z = (log_y_expanded - mus) / sigmas
        log_component = (
            -log_y_expanded
            - sigmas.log()
            - 0.5 * math.log(2 * math.pi)
            - 0.5 * z ** 2
        )
        log_mixture = torch.logsumexp(log_component + mix_weights.log(), dim=-1)
        log_p_positive = torch.log(1 - pi_zero + 1e-8) + log_mixture

        return zero_mask * log_p_zero + (1 - zero_mask) * log_p_positive

    def sample(self, params: dict[str, torch.Tensor], n_samples: int = 1000) -> torch.Tensor:
        """Draw samples from the predicted distribution."""
        batch_size = params["pi_zero"].shape[0]
        pi_zero = params["pi_zero"].squeeze(-1)
        mix_weights = params["mix_weights"]
        mus = params["mus"]
        sigmas = params["sigmas"]

        is_nonzero = torch.bernoulli(
            (1 - pi_zero).unsqueeze(-1).expand(-1, n_samples)
        )

        component_idx = torch.multinomial(
            mix_weights.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, self.n_components),
            1,
        ).reshape(batch_size, n_samples)

        selected_mu = torch.gather(
            mus.unsqueeze(1).expand(-1, n_samples, -1), 2, component_idx.unsqueeze(-1)
        ).squeeze(-1)
        selected_sigma = torch.gather(
            sigmas.unsqueeze(1).expand(-1, n_samples, -1), 2, component_idx.unsqueeze(-1)
        ).squeeze(-1)

        z = torch.randn_like(selected_mu)
        log_normal_samples = torch.exp(selected_mu + selected_sigma * z)
        samples = is_nonzero * log_normal_samples

        return torch.round(samples).clamp(min=0)

    def cdf(self, params: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """CDF at y, for CRPS evaluation."""
        pi_zero = params["pi_zero"].squeeze(-1)
        mix_weights = params["mix_weights"]
        mus = params["mus"]
        sigmas = params["sigmas"]

        log_y = torch.log(y.clamp(min=1e-6))
        z = (log_y.unsqueeze(-1) - mus) / sigmas
        component_cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        mixture_cdf = (mix_weights * component_cdf).sum(dim=-1)

        return (pi_zero + (1 - pi_zero) * mixture_cdf).clamp(0, 1)
