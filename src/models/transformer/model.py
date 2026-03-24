"""
Conv-Transformer for probabilistic conflict fatality forecasting.

Architecture:
    Causal Conv front-end  ->  Patch Embedding  ->  Rotary-PE Transformer
    ->  Hurdle-Student-t distribution head

Differences from a pure transformer (Model 2):
    - 2-layer causal conv extracts local escalation patterns before attention
    - 3-month patch tokenisation (quarter-level, not month-level)
    - Rotary positional encoding (relative, not absolute)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common.distribution_heads import HurdleStudentT


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = F.pad(x, (self.pad, 0))
        out = self.conv(out)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.drop(out)
        return out + res


class CausalConvFrontEnd(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, n_layers: int,
                 kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = in_features if i == 0 else hidden_dim
            layers.append(CausalConvBlock(in_ch, hidden_dim, kernel_size, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(1, 2)
        out = self.net(out)
        return out.transpose(1, 2)



class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, in_dim: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_dim * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert T % self.patch_size == 0
        n_patches = T // self.patch_size
        x = x.reshape(B, n_patches, self.patch_size * D)
        return self.norm(self.proj(x))



class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_head: int, max_len: int = 512):
        super().__init__()
        assert d_head % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)
        if T > self.cos_cached.size(0):
            self._build_cache(T)
        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)



class RoPEMultiHeadAttention(nn.Module):
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
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



class ConvTransformer(nn.Module):
    """Causal-Conv + Patch-Transformer with Hurdle-Student-t output."""

    def __init__(
        self,
        n_features: int,
        window_size: int = 24,
        patch_size: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 4,
        n_conv_layers: int = 2,
        conv_kernel_size: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv_frontend = CausalConvFrontEnd(
            n_features, d_model, n_conv_layers, conv_kernel_size, dropout,
        )
        self.patch_embed = PatchEmbedding(patch_size, d_model, d_model)
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.dist_head = HurdleStudentT(d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_frontend(x)
        h = self.patch_embed(h)
        h = self.transformer(h)
        h = self.final_norm(h)
        return h[:, -1, :]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return self.dist_head(h)

    def predict(self, x: torch.Tensor, n_samples: int = 1000) -> dict[str, torch.Tensor]:
        """Generate predictions with uncertainty via sampling."""
        self.eval()
        with torch.no_grad():
            params = self.forward(x)
            samples = self.dist_head.sample(params, n_samples)
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
                "p_conflict": torch.sigmoid(params["conflict_logit"]),
            }
