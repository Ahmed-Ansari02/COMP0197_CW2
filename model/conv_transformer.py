"""Conv-Transformer hybrid for fatality prediction.

Architecture: Causal-Conv front-end  ->  Patch embedding
              ->  Rotary-PE Transformer encoder  ->  Hurdle-Student-t head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Causal Conv Block
# ---------------------------------------------------------------------------

class CausalConvBlock(nn.Module):
    """Conv1d with left-padding so output depends only on current & past steps."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        res = self.residual(x)
        out = F.pad(x, (self.pad, 0))  # causal left-pad
        out = self.conv(out)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.drop(out)
        return out + res


class CausalConvFrontEnd(nn.Module):
    """Stack of CausalConvBlocks: projects input features to hidden_dim."""

    def __init__(self, in_features: int, hidden_dim: int, n_layers: int,
                 kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = in_features if i == 0 else hidden_dim
            layers.append(CausalConvBlock(in_ch, hidden_dim, kernel_size, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> (B, T, hidden_dim)"""
        out = x.transpose(1, 2)       # (B, F, T)
        out = self.net(out)            # (B, hidden_dim, T)
        return out.transpose(1, 2)     # (B, T, hidden_dim)


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Reshape sequence into non-overlapping patches and project to d_model."""

    def __init__(self, patch_size: int, in_dim: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_dim * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T/P, d_model)"""
        B, T, D = x.shape
        assert T % self.patch_size == 0, (
            f"Sequence length {T} not divisible by patch_size {self.patch_size}"
        )
        n_patches = T // self.patch_size
        x = x.reshape(B, n_patches, self.patch_size * D)
        x = self.proj(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Rotary Positional Encoding (RoPE)
# ---------------------------------------------------------------------------

class RotaryPositionalEncoding(nn.Module):
    """Applies rotary embeddings to query and key tensors."""

    def __init__(self, d_head: int, max_len: int = 512):
        super().__init__()
        assert d_head % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # (T, d_head/2)
        cos_cached = freqs.cos()                        # (T, d_head/2)
        sin_cached = freqs.sin()                        # (T, d_head/2)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_heads, T, d_head) -> rotated x"""
        T = x.size(2)
        if T > self.cos_cached.size(0):
            self._build_cache(T)
        cos = self.cos_cached[:T]   # (T, d_head/2)
        sin = self.sin_cached[:T]

        x1, x2 = x[..., ::2], x[..., 1::2]            # split pairs
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)
        return rotated.flatten(-2)                      # merge pairs back


# ---------------------------------------------------------------------------
# Transformer encoder layer with RoPE
# ---------------------------------------------------------------------------

class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with rotary positional encoding on Q and K."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.rope = RotaryPositionalEncoding(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RoPE attention."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

class HurdleStudentTHead(nn.Module):
    """Two-part hurdle output:
    - conflict_logit  -> P(fatalities > 0) via sigmoid
    - Student-t params (mu, sigma, nu) over log1p(fatalities) for non-zero cases
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conflict_head = nn.Linear(d_model, 1)
        self.severity_head = nn.Linear(d_model, 3)  # mu, pre_sigma, pre_nu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor]:
        """x: (B, d_model) -> conflict_logit, mu, sigma, nu"""
        h = self.shared(x)
        conflict_logit = self.conflict_head(h).squeeze(-1)          # (B,)
        severity = self.severity_head(h)                             # (B, 3)
        mu = severity[:, 0]                                          # (B,)
        sigma = F.softplus(severity[:, 1]) + 1e-4                    # (B,) > 0
        nu = F.softplus(severity[:, 2]) + 2.0                        # (B,) > 2 for finite variance
        return conflict_logit, mu, sigma, nu


class DeterministicHead(nn.Module):
    """Simple regression head for the MSE baseline."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ConvTransformer(nn.Module):
    """Causal-Conv + Patch-Transformer with Hurdle-Student-t output."""

    def __init__(
        self,
        n_features: int,
        window_size: int = 36,
        patch_size: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 4,
        n_conv_layers: int = 2,
        conv_kernel_size: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        baseline: bool = False,
    ):
        super().__init__()
        self.baseline = baseline

        # 1. Causal conv front-end
        self.conv_frontend = CausalConvFrontEnd(
            n_features, d_model, n_conv_layers, conv_kernel_size, dropout,
        )

        # 2. Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, d_model, d_model)

        # 3. Transformer encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 4. Output head
        if baseline:
            self.head = DeterministicHead(d_model, dropout)
        else:
            self.head = HurdleStudentTHead(d_model, dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> (B, d_model) pooled representation."""
        h = self.conv_frontend(x)       # (B, T, d_model)
        h = self.patch_embed(h)          # (B, n_patches, d_model)
        h = self.transformer(h)          # (B, n_patches, d_model)
        h = self.final_norm(h)
        return h[:, -1, :]               # last patch

    def forward(self, x: torch.Tensor):
        """Returns outputs depending on mode:
        - baseline: predicted log1p(fatalities)
        - hurdle:   (conflict_logit, mu, sigma, nu)
        """
        h = self.encode(x)
        return self.head(h)
