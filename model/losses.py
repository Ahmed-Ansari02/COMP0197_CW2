"""Loss functions for the Conv-Transformer fatality model.

Hurdle loss  = weighted BCE (conflict classifier)
             + Student-t NLL in log1p space (severity regressor, non-zero only)
"""

import math
import torch
import torch.nn.functional as F


def _student_t_log_prob(
    y_log: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    """Log-probability of the Student-t distribution.

    Parameters
    ----------
    y_log  : log1p(raw_count) — the observation in log-space
    mu     : location
    sigma  : scale (> 0)
    nu     : degrees of freedom (> 0, we enforce > 2)
    """
    z = (y_log - mu) / sigma
    log_prob = (
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(nu * math.pi)
        - torch.log(sigma)
        - ((nu + 1) / 2) * torch.log1p(z * z / nu)
    )
    return log_prob


def hurdle_loss(
    conflict_logit: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor,
    y_count: torch.Tensor,
    y_binary: torch.Tensor,
    pos_weight: float = 5.1,
) -> torch.Tensor:
    """Combined hurdle loss.

    Parameters
    ----------
    conflict_logit : (B,)  raw logit for P(fatalities > 0)
    mu, sigma, nu  : (B,)  Student-t params in log1p space
    y_count        : (B,)  raw fatality count
    y_binary       : (B,)  1 if count > 0, else 0
    pos_weight     : scalar weight for positive class in BCE

    Returns
    -------
    total_loss, bce_loss, nll_loss  (all scalar)
    """
    pw = torch.tensor([pos_weight], device=conflict_logit.device, dtype=conflict_logit.dtype)
    bce = F.binary_cross_entropy_with_logits(
        conflict_logit, y_binary, pos_weight=pw,
    )

    # Student-t NLL only on non-zero samples
    mask = y_binary > 0.5
    if mask.any():
        y_log = torch.log1p(y_count[mask])
        nll = -_student_t_log_prob(y_log, mu[mask], sigma[mask], nu[mask]).mean()
    else:
        nll = torch.tensor(0.0, device=conflict_logit.device)

    total = bce + nll
    return total, bce, nll


def mse_baseline_loss(
    y_pred: torch.Tensor,
    y_count: torch.Tensor,
) -> torch.Tensor:
    """MSE loss on log1p(fatalities) for the deterministic baseline."""
    y_log = torch.log1p(y_count)
    return F.mse_loss(y_pred, y_log)
