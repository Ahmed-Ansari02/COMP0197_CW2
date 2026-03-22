"""
model.py — Bayesian LSTM with ZINB output head
Epistemic uncertainty via MC dropout (active at train AND inference)
Aleatoric uncertainty via Zero-Inflated Negative Binomial parameters

Architecture:
    dynamic [B, T, n_dyn] → LSTM encoder → hidden [B, hidden_dim]
    static  [B, n_sta]    → static MLP   → [B, static_dim]
    concat [B, hidden_dim + static_dim] → fusion MLP → ZINB head (mu, alpha, pi)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ZINB Distribution helpers ─────────────────────────────────────────────────

def zinb_nll(y, mu, alpha, pi):
    """
    Zero-Inflated Negative Binomial negative log-likelihood.

    P(Y=0)   = pi + (1-pi) * NegBin(0; mu, alpha)
    P(Y=k>0) = (1-pi) * NegBin(k; mu, alpha)

    NegBin parameterised by mean mu and dispersion alpha:
        r = 1/alpha
        p = r / (r + mu)

    Args:
        y     : [B] observed counts (log1p scale — we work in log space)
        mu    : [B] predicted mean (softplus, log space)
        alpha : [B] dispersion     (softplus, > 0)
        pi    : [B] zero-inflation probability (sigmoid)

    Returns:
        nll : [B] per-sample negative log-likelihood
    """
    eps = 1e-8

    # Work in count space for NegBin — back-transform from log1p
    y_count = torch.expm1(y).clamp(min=0)
    mu_count = torch.expm1(mu).clamp(min=eps)

    r = (1.0 / alpha.clamp(min=eps))          # NegBin r parameter
    p = r / (r + mu_count)                     # success probability

    # log NegBin(0; mu, alpha) = r * log(p)
    log_nb_zero = r * torch.log(p.clamp(min=eps))

    # log NegBin(k; mu, alpha) via lgamma
    log_nb_k = (
        torch.lgamma(r + y_count)
        - torch.lgamma(r)
        - torch.lgamma(y_count + 1)
        + r * torch.log(p.clamp(min=eps))
        + y_count * torch.log((1 - p).clamp(min=eps))
    )

    # Zero component
    log_pi     = torch.log(pi.clamp(min=eps))
    log_1mpi   = torch.log((1 - pi).clamp(min=eps))

    # P(Y=0) = pi + (1-pi)*NegBin(0)
    log_prob_zero    = torch.logaddexp(log_pi, log_1mpi + log_nb_zero)
    # P(Y=k>0) = (1-pi)*NegBin(k)
    log_prob_nonzero = log_1mpi + log_nb_k

    zero_mask = (y_count == 0)
    nll = torch.where(zero_mask, -log_prob_zero, -log_prob_nonzero)

    return nll


# ── Building blocks ───────────────────────────────────────────────────────────

class MCDropout(nn.Module):
    """Dropout that stays active at inference time for MC sampling."""
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        # Always active — ignore model.eval()
        return F.dropout(x, p=self.p, training=True)


class StaticMLP(nn.Module):
    """Small MLP to encode static country features."""
    def __init__(self, n_static: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_static, out_dim * 2),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            MCDropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FusionMLP(nn.Module):
    """MLP that takes concatenated LSTM + static encodings → ZINB params."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            MCDropout(dropout),
        )
        self.out_dim = hidden_dim // 2

    def forward(self, x):
        return self.net(x)


class ZINBHead(nn.Module):
    """
    Three-headed output layer for ZINB distribution.
        mu    : softplus  → predicted mean fatalities (log1p scale)
        alpha : softplus  → dispersion (larger = more overdispersed)
        pi    : sigmoid   → P(structural zero)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.mu_head    = nn.Linear(in_dim, 1)
        self.alpha_head = nn.Linear(in_dim, 1)
        self.pi_head    = nn.Linear(in_dim, 1)

    def forward(self, x):
        mu    = F.softplus(self.mu_head(x)).squeeze(-1)
        alpha = F.softplus(self.alpha_head(x)).squeeze(-1) + 1e-4
        pi    = torch.sigmoid(self.pi_head(x)).squeeze(-1)
        return mu, alpha, pi


# ── Main model ────────────────────────────────────────────────────────────────

class BayesianLSTM(nn.Module):
    """
    Bayesian LSTM for conflict fatality prediction.

    Args:
        n_dynamic   : number of dynamic (monthly) features
        n_static    : number of static (country-level) features
        hidden_dim  : LSTM hidden size
        n_layers    : number of LSTM layers
        static_dim  : encoded size of static features
        fusion_dim  : hidden size of fusion MLP
        dropout     : MC dropout probability (applied everywhere)
    """

    def __init__(
        self,
        n_dynamic:  int,
        n_static:   int,
        hidden_dim: int = 128,
        n_layers:   int = 2,
        static_dim: int = 32,
        fusion_dim: int = 128,
        dropout:    float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.dropout    = dropout

        # LSTM encoder — dropout between layers (not on last layer output)
        self.lstm = nn.LSTM(
            input_size=n_dynamic,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # MC dropout on LSTM output
        self.lstm_dropout = MCDropout(dropout)

        # Static encoder
        self.static_mlp = StaticMLP(n_static, static_dim, dropout)

        # Fusion MLP
        self.fusion = FusionMLP(hidden_dim + static_dim, fusion_dim, dropout)

        # ZINB output head
        self.zinb_head = ZINBHead(fusion_dim // 2)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    
        # Bias pi head toward predicting zeros
        # sigmoid(3.0) ≈ 0.95 — start assuming most months are zero-conflict
        nn.init.constant_(self.zinb_head.pi_head.bias, 3.0)
        
        # Bias mu head toward low values
        nn.init.constant_(self.zinb_head.mu_head.bias, -1.0)

    def forward(self, dynamic, static):
        """
        Args:
            dynamic : [B, T, n_dynamic]
            static  : [B, n_static]
        Returns:
            mu, alpha, pi : each [B]
        """
        # LSTM — take final hidden state
        lstm_out, (h_n, _) = self.lstm(dynamic)
        # h_n: [n_layers, B, hidden_dim] — take last layer
        h_last = h_n[-1]                          # [B, hidden_dim]
        h_last = self.lstm_dropout(h_last)

        # Static encoding
        s = self.static_mlp(static)               # [B, static_dim]

        # Fuse
        fused = torch.cat([h_last, s], dim=-1)    # [B, hidden_dim + static_dim]
        fused = self.fusion(fused)                 # [B, fusion_dim//2]

        # ZINB parameters
        mu, alpha, pi = self.zinb_head(fused)

        return mu, alpha, pi

    def predict_with_uncertainty(self, dynamic, static, n_samples: int = 50):
        """
        MC dropout inference — run n_samples forward passes.

        Returns:
            mu_samples    : [n_samples, B]  predicted means per pass
            alpha_samples : [n_samples, B]  dispersion per pass
            pi_samples    : [n_samples, B]  zero-prob per pass
            epistemic_var : [B]  variance of mu across MC samples
            aleatoric_var : [B]  mean ZINB variance across MC samples
            mu_mean       : [B]  mean prediction
        """
        self.train()  # ensure dropout is active

        mu_list, alpha_list, pi_list = [], [], []

        with torch.no_grad():
            for _ in range(n_samples):
                mu, alpha, pi = self(dynamic, static)
                mu_list.append(mu)
                alpha_list.append(alpha)
                pi_list.append(pi)

        mu_samples    = torch.stack(mu_list,    dim=0)   # [S, B]
        alpha_samples = torch.stack(alpha_list, dim=0)   # [S, B]
        pi_samples    = torch.stack(pi_list,    dim=0)   # [S, B]

        # Epistemic: variance of predicted mean across MC passes
        epistemic_var = mu_samples.var(dim=0)            # [B]

        aleatoric_var = alpha_samples.mean(dim=0)

        mu_mean = mu_samples.mean(dim=0)                 # [B]

        return mu_samples, alpha_samples, pi_samples, epistemic_var, aleatoric_var, mu_mean


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B, T, n_dyn, n_sta = 256, 24, 93, 7

    model = BayesianLSTM(
        n_dynamic=n_dyn,
        n_static=n_sta,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    dyn = torch.randn(B, T, n_dyn).to(device)
    sta = torch.randn(B, n_sta).to(device)
    tgt = torch.rand(B).to(device)

    # Forward pass
    mu, alpha, pi = model(dyn, sta)
    print(f"\nForward pass:")
    print(f"  mu    : {mu.shape}  min={mu.min():.3f}  max={mu.max():.3f}")
    print(f"  alpha : {alpha.shape}  min={alpha.min():.4f}  max={alpha.max():.3f}")
    print(f"  pi    : {pi.shape}  min={pi.min():.3f}  max={pi.max():.3f}")

    # Loss
    nll = zinb_nll(tgt, mu, alpha, pi)
    loss = nll.mean()
    print(f"\nZINB NLL loss: {loss.item():.4f}")
    loss.backward()
    print("Backward pass: OK")

    # MC uncertainty
    mu_s, alpha_s, pi_s, epi_var, ale_var, mu_mean = model.predict_with_uncertainty(dyn, sta, n_samples=10)
    print(f"\nMC uncertainty (10 samples):")
    print(f"  mu_samples    : {mu_s.shape}")
    print(f"  epistemic_var : mean={epi_var.mean():.4f}")
    print(f"  aleatoric_var : mean={ale_var.mean():.4f}")