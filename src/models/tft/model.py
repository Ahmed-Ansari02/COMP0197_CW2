"""
architecture:
- Linear projection (replacing my prev VariableSelectionNetwork)
- 1-layer LSTM encoder (replaces 2-layer + GRN)
- Multi-head self-attention with skip connection
- ZILNM distribution head

"""

import torch
import torch.nn as nn

from src.models.common.distribution_heads import ZeroInflatedLogNormalMixture


class ConflictForecaster(nn.Module):
    """
    input projection → LSTM → attention → ZILNM head.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_lstm_layers: int = 1,
        n_attention_heads: int = 2,
        n_mixture_components: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Simple linear projection (replaces VSN)
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
        )

        # Multi-head attention with simple skip connection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dist_head = ZeroInflatedLogNormalMixture(hidden_dim, n_mixture_components)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, n_features]

        Returns:
            dict with ZILNM distribution parameters
        """
        # Project input features
        h = self.input_proj(x)  # [batch, seq, hidden]

        # LSTM
        lstm_out, _ = self.lstm(h)  # [batch, seq, hidden]

        # Self-attention + residual
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        h = self.attn_norm(attn_out + lstm_out)  # simple add & norm

        # Last timestep → distribution params
        h_last = h[:, -1, :]
        h_out = self.output_proj(h_last)
        dist_params = self.dist_head(h_out)

        if return_attention:
            dist_params["attention_weights"] = attn_weights

        return dist_params

    def predict(
        self,
        x: torch.Tensor,
        n_samples: int = 1000,
    ) -> dict[str, torch.Tensor]:
        """Generate predictions with uncertainty estimates."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            samples = self.dist_head.sample(output, n_samples)

            return {
                "samples": samples,
                "mean": samples.mean(dim=1),
                "median": samples.median(dim=1).values,
                "p05": samples.quantile(0.05, dim=1),
                "p10": samples.quantile(0.10, dim=1),
                "p25": samples.quantile(0.25, dim=1),
                "p75": samples.quantile(0.75, dim=1),
                "p90": samples.quantile(0.90, dim=1),
                "p95": samples.quantile(0.95, dim=1),
                "pi_zero": output["pi_zero"].squeeze(-1),
            }
