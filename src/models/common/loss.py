"""
Spike-aware NLL loss for conflict forecasting.

Upweights rare high-fatality events so the model doesn't
ignore them in favour of predicting zeros everywhere.
"""

import torch
import torch.nn as nn


class ConflictForecastLoss(nn.Module):
    """
    NLL loss with spike weighting.

    Observations with y > spike_threshold get multiplied by spike_loss_multiplier
    to ensure the model learns to forecast rare extreme events.
    """

    def __init__(
        self,
        spike_threshold: float = 500,
        spike_loss_multiplier: float = 5.0,
    ):
        super().__init__()
        self.spike_threshold = spike_threshold
        self.spike_loss_multiplier = spike_loss_multiplier

    def forward(
        self,
        dist_params: dict[str, torch.Tensor],
        y: torch.Tensor,
        dist_head,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            dist_params: output of ZeroInflatedLogNormalMixture.forward()
            y: true fatalities [batch]
            dist_head: ZeroInflatedLogNormalMixture module

        Returns:
            dict with total loss and diagnostics
        """
        log_probs = dist_head.log_prob(dist_params, y)
        nll = -log_probs

        # Spike weighting
        spike_weight = torch.ones_like(y)
        spike_weight[y > self.spike_threshold] = self.spike_loss_multiplier
        weighted_nll = (nll * spike_weight).mean()

        return {
            "loss": weighted_nll,
            "nll": weighted_nll.detach(),
            "mean_log_prob": log_probs.mean().detach(),
        }
