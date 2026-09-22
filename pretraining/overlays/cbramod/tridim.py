"""TriDim: tri-axis EEG encoding with multi-level readout.

The model normalizes each channel along time, extracts temporal patches,
and projects the channel, within-patch and across-patch axes into learned
basis dimensions. In the implementation these axes are named C, K and T;
they correspond to C, S and L in the paper.

Each encoder block applies parallel cross-axis attention and axis-specific
feed-forward networks, with learned softmax fusion and residual LayerScale.
Attention weights are shared across the two attention operations within
each branch. DropPath rates are configurable for each attention branch and
the feed-forward path. The optional multi-level readout pools each encoder
layer separately and fuses its representation before classification.

Key options: use_input_norm, use_multi_level_readout, drop_path_c,
drop_path_k, drop_path_t, drop_path_mlp, and drop_path_schedule.

The pretraining integration also supports an optional coordinate-based
channel adapter via use_channel_adapter and channel coordinate settings.
"""

import csv
import math
import os
import random
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tech.layers.Augmentation import get_augmentation
except Exception:
    try:
        from layers.Augmentation import get_augmentation
    except Exception:
        def get_augmentation(spec):
            return nn.Identity()


def _get_config(configs, name: str, default=None):
    if isinstance(configs, dict):
        value = configs.get(name, default)
    else:
        value = getattr(configs, name, default)
    # Treat argparse attributes set to None as unspecified.
    return default if value is None else value


# =============================================================================
# Channel coordinate utilities 
# =============================================================================

def load_channel_coordinate_csv(
    csv_path: str,
    expected_channels: Optional[int] = None,
    channel_order: Optional[Sequence[str]] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Sequence[str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Channel coordinate CSV not found: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"channel", "x", "y", "z"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"CSV {csv_path} must contain columns {sorted(required)}, "
                f"missing {sorted(missing)}"
            )
        rows = []
        for row in reader:
            rows.append(
                (str(row["channel"]).strip(),
                 [float(row["x"]), float(row["y"]), float(row["z"])])
            )
    if channel_order is not None:
        lookup = {name: xyz for name, xyz in rows}
        names = list(channel_order)
        missing = [name for name in names if name not in lookup]
        if missing:
            raise ValueError(f"CSV {csv_path} missing requested channels: {missing}")
        coords = [lookup[name] for name in names]
    else:
        names = [name for name, _ in rows]
        coords = [xyz for _, xyz in rows]
    coords = torch.tensor(coords, dtype=dtype)
    if expected_channels is not None and coords.size(0) != int(expected_channels):
        raise ValueError(
            f"Expected {expected_channels} channels, got {coords.size(0)} from {csv_path}"
        )
    return coords, names


def fibonacci_sphere(num_points: int, device=None, dtype=None) -> torch.Tensor:
    if num_points <= 1:
        return torch.zeros(num_points, 3, device=device, dtype=dtype)
    indices = torch.arange(num_points, device=device, dtype=torch.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (2.0 * indices / (num_points - 1))
    radius = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = phi * indices
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius
    coords = torch.stack([x, y, z], dim=-1)
    if dtype is not None:
        coords = coords.to(dtype=dtype)
    return coords


def pairwise_rbf_logits(
    target_coords: torch.Tensor,
    source_coords: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    rel = target_coords.unsqueeze(-2) - source_coords.unsqueeze(-3)
    dist2 = (rel ** 2).sum(dim=-1)
    sigma2 = max(float(sigma), 1e-6) ** 2
    return -dist2 / (2.0 * sigma2)


def masked_softmax(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(dtype=logits.dtype)
    very_neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(mask == 0, very_neg)
    weights = torch.softmax(logits, dim=dim)
    weights = weights * mask
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(eps)
    return weights / denom


class CoordinateResidualMapper(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        target_coords: torch.Tensor,
        source_coords: torch.Tensor,
    ) -> torch.Tensor:
        if target_coords.ndim == 2:
            target_coords = target_coords.unsqueeze(0)
        if source_coords.ndim == 2:
            source_coords = source_coords.unsqueeze(0)
        if target_coords.size(0) != source_coords.size(0):
            if target_coords.size(0) == 1:
                target_coords = target_coords.expand(source_coords.size(0), -1, -1)
            elif source_coords.size(0) == 1:
                source_coords = source_coords.expand(target_coords.size(0), -1, -1)
            else:
                raise ValueError(
                    "Batch mismatch between target_coords and source_coords"
                )
        rel = target_coords.unsqueeze(2) - source_coords.unsqueeze(1)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        return self.net(torch.cat([rel, dist], dim=-1)).squeeze(-1)


class ChannelAdapter(nn.Module):
    def __init__(
        self,
        canonical_channels: int = 64,
        use_prior: bool = True,
        sigma: float = 0.35,
        residual_hidden_dim: int = 32,
        residual_scale_init: float = 0.10,
        canonical_coords: Optional[torch.Tensor] = None,
        canonical_channel_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.canonical_channels = int(canonical_channels)
        self.use_prior = bool(use_prior)
        self.sigma = float(sigma)
        self.residual_mapper = CoordinateResidualMapper(hidden_dim=residual_hidden_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        if canonical_coords is None:
            canonical_coords = fibonacci_sphere(self.canonical_channels)
        self.register_buffer("canonical_coords", canonical_coords.float(), persistent=True)
        self.canonical_channel_names = (
            list(canonical_channel_names) if canonical_channel_names is not None else None
        )

    def _identity_map(self, cin: int, device, dtype) -> torch.Tensor:
        eye = torch.eye(cin, device=device, dtype=dtype)
        if cin == self.canonical_channels:
            return eye
        out = torch.zeros(self.canonical_channels, cin, device=device, dtype=dtype)
        copy_n = min(cin, self.canonical_channels)
        out[:copy_n, :copy_n] = eye[:copy_n, :copy_n]
        return out

    def _compute_weights(
        self,
        channel_coords: torch.Tensor,
        channel_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if channel_coords.ndim == 2:
            channel_coords = channel_coords.unsqueeze(0)
        batch_size = channel_coords.size(0)
        target_coords = (
            self.canonical_coords.to(device=channel_coords.device, dtype=dtype)
            .unsqueeze(0).expand(batch_size, -1, -1)
        )
        geom_logits = pairwise_rbf_logits(
            target_coords, channel_coords.to(dtype=dtype), sigma=self.sigma,
        )
        learned_logits = self.residual_mapper(
            target_coords, channel_coords.to(dtype=dtype),
        )
        mask = None
        if channel_mask is not None:
            if channel_mask.ndim == 1:
                channel_mask = channel_mask.unsqueeze(0)
            mask = channel_mask.unsqueeze(1).expand(-1, self.canonical_channels, -1)
        if self.use_prior:
            geom_weights = masked_softmax(geom_logits, mask, dim=-1)
            learned_weights = masked_softmax(learned_logits, mask, dim=-1)
            weights = geom_weights + self.residual_scale * learned_weights
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            weights = masked_softmax(learned_logits, mask, dim=-1)
        return weights

    def forward(
        self,
        x: torch.Tensor,
        channel_coords: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, cin, _ = x.shape
        if channel_coords is None:
            if cin != self.canonical_channels:
                raise ValueError(
                    "channel_coords is required when input channel count "
                    "differs from canonical_channels"
                )
            weights = (
                self._identity_map(cin, x.device, x.dtype)
                .unsqueeze(0).expand(batch_size, -1, -1)
            )
        else:
            weights = self._compute_weights(channel_coords, channel_mask, x.dtype)
            if weights.size(0) == 1 and batch_size > 1:
                weights = weights.expand(batch_size, -1, -1)
        return torch.einsum("boc,bct->bot", weights.to(dtype=x.dtype), x)


# =============================================================================
# Cross-subject normalisation at the input
# =============================================================================

class InstanceTimeNorm(nn.Module):
    """
    Per-sample per-channel normalisation along the time axis.

    Input:
        x: [B, C, L]
    Output:
        normalised x with the same shape.

    Removes subject-specific DC offset and amplitude scaling at the input.
    No learnable parameters; safe under Domain Generalisation.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


# =============================================================================
# DropPath (per-sample stochastic depth) — needed for DSS
# =============================================================================

class DropPath(nn.Module):
    """Per-sample stochastic depth, scaled by 1 / (1 - p) during training."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


# =============================================================================
# Axis projections 
# =============================================================================

class LinearChannelProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LinearKProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.transpose(2, 3)
        out = self.proj(x_perm)
        return out.transpose(2, 3).contiguous()


class ConvTProjection(nn.Module):
    """Conv1d + AdaptiveAvgPool1d along the global temporal patch axis T.

    NOTE (cleanup TODO): the internal Conv1d is 1→1 channel with kernel_size=5
    shared across all (B, C, K) sequences — only ~6 parameters of expressive
    capacity in total. A simpler design would replace this whole class with
    `nn.AdaptiveAvgPool1d(out_t_dim)` directly. Kept here for now to avoid
    silent behaviour change on existing experiments; replace once an ablation
    confirms it does not hurt.
    """

    def __init__(
        self,
        out_t_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()
        self.out_t_dim = int(out_t_dim)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = dilation * (kernel_size - 1) // 2
        self.proj = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=1,
                kernel_size=kernel_size, padding=padding,
                dilation=dilation, bias=True,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(self.out_t_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, k, t = x.shape
        y = x.reshape(b * c * k, 1, t)
        y = self.proj(y)
        y = self.pool(y)
        y = y.reshape(b, c, k, self.out_t_dim)
        return y


# =============================================================================
# Tri-axis attention pooling readout 
# =============================================================================

class AttentionPool1D(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_logits = self.score(tokens).squeeze(-1)        # [B, L]
        attn_weights = torch.softmax(attn_logits, dim=-1)   # [B, L]
        pooled = torch.sum(tokens * attn_weights.unsqueeze(-1), dim=1)
        return pooled


class TriAxisAttentionPoolingHead(nn.Module):
    """
    Tri-axis attention pooling head.

    In "classifier" mode (default), pools [B, D, K, T] along each of the
    three axes (C / K / T), fuses the three pooled vectors with a learnable
    softmax, and applies a linear classifier to produce logits.

    In "feature" mode (`return_features=True`), the classifier head is
    skipped — the module returns the fused embed_dim vector instead. This
    mode is used by MultiLevelTriAxisReadout to share one pooling head
    per encoder layer without a classifier per layer.
    """

    def __init__(
        self,
        channel_dim: int,
        k_dim: int,
        t_dim: int,
        embed_dim: int,
        num_class: int,
        dropout: float = 0.0,
        return_features: bool = False,
    ):
        super().__init__()
        self.channel_dim = int(channel_dim)
        self.k_dim = int(k_dim)
        self.t_dim = int(t_dim)
        self.embed_dim = int(embed_dim)
        self.num_class = int(num_class)
        self.return_features = bool(return_features)

        self.c_proj = nn.Linear(self.k_dim * self.t_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.channel_dim * self.t_dim, self.embed_dim)
        self.t_proj = nn.Linear(self.channel_dim * self.k_dim, self.embed_dim)

        self.c_pos = nn.Parameter(torch.zeros(1, self.channel_dim, self.embed_dim))
        self.k_pos = nn.Parameter(torch.zeros(1, self.k_dim, self.embed_dim))
        self.t_pos = nn.Parameter(torch.zeros(1, self.t_dim, self.embed_dim))

        self.c_pool = AttentionPool1D(self.embed_dim, dropout=dropout)
        self.k_pool = AttentionPool1D(self.embed_dim, dropout=dropout)
        self.t_pool = AttentionPool1D(self.embed_dim, dropout=dropout)

        self.axis_fusion_logits = nn.Parameter(torch.zeros(3))

        # Classifier head is only built when not in feature mode.
        if not self.return_features:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, self.num_class),
            )
        else:
            self.classifier = None

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the tri-axis pooling + fusion and return the embed_dim vector.

        This is the shared sub-routine used by both forward() and
        MultiLevelTriAxisReadout.
        """
        b, c, k, t = x.shape
        if c != self.channel_dim or k != self.k_dim or t != self.t_dim:
            raise ValueError(
                f"Expected x shape [B,{self.channel_dim},{self.k_dim},{self.t_dim}], "
                f"but got {tuple(x.shape)}"
            )

        c_tokens = x.reshape(b, c, k * t)
        c_tokens = self.c_proj(c_tokens) + self.c_pos

        k_tokens = x.permute(0, 2, 1, 3).contiguous().reshape(b, k, c * t)
        k_tokens = self.k_proj(k_tokens) + self.k_pos

        t_tokens = x.permute(0, 3, 1, 2).contiguous().reshape(b, t, c * k)
        t_tokens = self.t_proj(t_tokens) + self.t_pos

        h_c = self.c_pool(c_tokens)
        h_k = self.k_pool(k_tokens)
        h_t = self.t_pool(t_tokens)

        axis_weights = torch.softmax(self.axis_fusion_logits, dim=0)
        h = (
            axis_weights[0] * h_c
            + axis_weights[1] * h_k
            + axis_weights[2] * h_t
        )
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool_features(x)
        if self.return_features:
            return h
        return self.classifier(h)


class MultiLevelTriAxisReadout(nn.Module):
    """
    TriDim multi-level tri-axis readout (DIP-inspired).

    Given the per-layer outputs from the encoder
        [layer_0_out, layer_1_out, ..., layer_{N-1}_out]
    each shaped [B, D, K, T], this readout:

        1. Runs an independent TriAxisAttentionPoolingHead (in feature mode)
           on each layer's output, producing a per-layer embed_dim vector.
        2. Fuses the per-layer vectors with a learnable softmax weight
           (n_layers entries, initialised to zero → uniform after softmax).
        3. Applies a single classifier head to the fused vector.

    Compared to using only the final layer:
      * Improves gradient flow to shallow layers (extra supervision signal).
      * Gives the classifier a multi-scale view of the encoder hierarchy.
      * Cheap: each per-layer pool reuses the well-tested
        TriAxisAttentionPoolingHead design.

    Compared to EEG-Deformer's Dense Information Purification:
      * Lightweight: no dense skip connections through the backbone, just
        late-stage pooling.
      * Fully decoupled from the backbone, which keeps the TriDim tri-axis
        core untouched.
    """

    def __init__(
        self,
        n_layers: int,
        channel_dim: int,
        k_dim: int,
        t_dim: int,
        embed_dim: int,
        num_class: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.embed_dim = int(embed_dim)
        self.num_class = int(num_class)

        # One pooling head per encoder layer, sharing the TriAxis design
        # but in feature mode (no per-layer classifier).
        self.layer_pools = nn.ModuleList([
            TriAxisAttentionPoolingHead(
                channel_dim=channel_dim,
                k_dim=k_dim,
                t_dim=t_dim,
                embed_dim=self.embed_dim,
                num_class=self.num_class,    # unused in feature mode
                dropout=dropout,
                return_features=True,
            )
            for _ in range(self.n_layers)
        ])

        # Learnable per-layer fusion weights (initialised to zero ->
        # softmax gives uniform 1/n_layers across all layers).
        self.layer_fusion_logits = nn.Parameter(torch.zeros(self.n_layers))

        # Single classifier head on top of the fused vector.
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.num_class),
        )

    def forward(self, layer_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(layer_outputs) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} layer outputs, got {len(layer_outputs)}"
            )

        # Per-layer pooling: each gives [B, embed_dim].
        pooled = [
            pool.pool_features(layer_out)
            for pool, layer_out in zip(self.layer_pools, layer_outputs)
        ]
        stacked = torch.stack(pooled, dim=1)             # [B, n_layers, embed_dim]

        weights = torch.softmax(self.layer_fusion_logits, dim=0)  # [n_layers]
        fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)     # [B, embed_dim]

        return self.classifier(fused)


# =============================================================================
# Building blocks for the encoder 
# =============================================================================

class AxisRMSNorm(nn.Module):
    """
    RMSNorm applied along a specific axis of a 4D tensor [B, C, K, T].

    For axis=a, normalises each element by the RMS of the values along
    axis `a` (per (other-axis) coordinate), and rescales by a learnable
    per-element weight on that axis.
    """

    def __init__(self, dim_size: int, axis: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim_size)))
        self.axis = int(axis)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.transpose(self.axis, -1).contiguous()
        rms = x_perm.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_perm = (x_perm / rms) * self.weight
        return x_perm.transpose(self.axis, -1).contiguous()


class AxisMLP(nn.Module):
    def __init__(self, dim_size: int, dropout: float = 0.0, expansion: int = 2):
        super().__init__()
        self.dim_size = int(dim_size)
        hidden = self.dim_size * int(expansion)
        self.net = nn.Sequential(
            nn.Linear(self.dim_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.dim_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        x_perm = x.transpose(dim, -1)
        original_shape = x_perm.shape
        out = self.net(x_perm.reshape(-1, self.dim_size)).reshape(original_shape)
        return out.transpose(dim, -1)


class AxisAttention(nn.Module):
    """Multi-head self-attention along an arbitrary attend-axis of a 4D tensor."""

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        if self.embed_dim % max(1, int(n_heads)) != 0:
            n_heads = 1
        self.n_heads = max(1, int(n_heads))
        self.head_dim = self.embed_dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.drop = nn.Dropout(dropout)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1,
        )
        out = (
            torch.matmul(self.drop(attn), v)
            .transpose(1, 2)
            .reshape(batch, seq_len, self.embed_dim)
        )
        return self.drop(self.out_proj(out))

    def forward(
        self,
        x: torch.Tensor,
        attend_dim: int,
        embed_axis: int,
    ) -> torch.Tensor:
        batch, c_dim, k_dim, t_dim = x.shape
        if embed_axis == 1 and attend_dim == 2:
            return (
                self._attention(
                    x.permute(0, 3, 2, 1).reshape(batch * t_dim, k_dim, c_dim)
                )
                .reshape(batch, t_dim, k_dim, c_dim)
                .permute(0, 3, 2, 1)
            )
        if embed_axis == 1 and attend_dim == 3:
            return (
                self._attention(
                    x.permute(0, 2, 3, 1).reshape(batch * k_dim, t_dim, c_dim)
                )
                .reshape(batch, k_dim, t_dim, c_dim)
                .permute(0, 3, 1, 2)
            )
        if embed_axis == 2 and attend_dim == 1:
            return (
                self._attention(
                    x.permute(0, 3, 1, 2).reshape(batch * t_dim, c_dim, k_dim)
                )
                .reshape(batch, t_dim, c_dim, k_dim)
                .permute(0, 2, 3, 1)
            )
        if embed_axis == 2 and attend_dim == 3:
            return (
                self._attention(
                    x.permute(0, 1, 3, 2).reshape(batch * c_dim, t_dim, k_dim)
                )
                .reshape(batch, c_dim, t_dim, k_dim)
                .permute(0, 1, 3, 2)
            )
        if embed_axis == 3 and attend_dim == 1:
            return (
                self._attention(
                    x.permute(0, 2, 1, 3).reshape(batch * k_dim, c_dim, t_dim)
                )
                .reshape(batch, k_dim, c_dim, t_dim)
                .permute(0, 2, 1, 3)
            )
        if embed_axis == 3 and attend_dim == 2:
            return self._attention(
                x.reshape(batch * c_dim, k_dim, t_dim)
            ).reshape(batch, c_dim, k_dim, t_dim)
        raise ValueError("Unsupported attention configuration")


# =============================================================================
# TriDim block — cross-axis attention with per-branch DropPath
# =============================================================================

class TriAxisMixerBlock(nn.Module):
    """
    TriDim pre-normalization + dual-residual + shared cross-axis attention
        + per-channel LayerScale + Dimension-Specific Stochastic Depth (DSS).

    Structure:

        # Attention path: 3 sub-branches in parallel, each with its own DropPath
        c_out = dp_c(0.5 * (attn_for_c(norm_c, attend=K) + attn_for_c(norm_c, attend=T)))
        k_out = dp_k(0.5 * (attn_for_k(norm_k, attend=C) + attn_for_k(norm_k, attend=T)))
        t_out = dp_t(0.5 * (attn_for_t(norm_t, attend=C) + attn_for_t(norm_t, attend=K)))
        attn_branch = w_attn ·-fused (c_out, k_out, t_out)
        x = x + γ_attn ⊙ attn_branch

        # MLP path: 3 sub-branches in parallel, single DropPath at the end
        c_out = channel_mlp(norm_c_mlp, dim=1)
        k_out = k_mlp(norm_k_mlp, dim=2)
        t_out = t_mlp(norm_t_mlp, dim=3)
        mlp_branch = dp_mlp(w_mlp ·-fused (c_out, k_out, t_out))
        x = x + γ_mlp ⊙ mlp_branch

    Stochastic-depth configuration:
        * Each attention sub-branch is gated by its own DropPath rate (DSS).
          Default: drop_path_c=0.25, drop_path_k=0.05, drop_path_t=0.15.
        * The MLP path is gated by a single DropPath rate (drop_path_mlp).
        * All four rates are configurable per block (used by the encoder
          to implement a linear DropPath schedule across layers).

    Setting all drop_path_* to zero disables stochastic depth.
    """

    def __init__(
        self,
        channel_dim: int,
        k_dim: int,
        t_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-2,
        drop_path_c: float = 0.0,
        drop_path_k: float = 0.0,
        drop_path_t: float = 0.0,
        drop_path_mlp: float = 0.0,
    ):
        super().__init__()
        self.channel_dim = int(channel_dim)
        self.k_dim = int(k_dim)
        self.t_dim = int(t_dim)

        # ---- Per-axis RMSNorm: one set for the attention path, one for MLP ----
        self.norm_c_attn = AxisRMSNorm(self.channel_dim, axis=1)
        self.norm_k_attn = AxisRMSNorm(self.k_dim,       axis=2)
        self.norm_t_attn = AxisRMSNorm(self.t_dim,       axis=3)
        self.norm_c_mlp = AxisRMSNorm(self.channel_dim, axis=1)
        self.norm_k_mlp = AxisRMSNorm(self.k_dim,       axis=2)
        self.norm_t_mlp = AxisRMSNorm(self.t_dim,       axis=3)

        # ---- Shared cross-axis attention: 3 modules, each used twice ----
        self.attn_for_c = AxisAttention(self.channel_dim, n_heads=n_heads, dropout=dropout)
        self.attn_for_k = AxisAttention(self.k_dim,       n_heads=n_heads, dropout=dropout)
        self.attn_for_t = AxisAttention(self.t_dim,       n_heads=n_heads, dropout=dropout)

        # ---- Axis-wise MLPs ----
        self.channel_mlp = AxisMLP(self.channel_dim, dropout=dropout)
        self.k_mlp       = AxisMLP(self.k_dim,       dropout=dropout)
        self.t_mlp       = AxisMLP(self.t_dim,       dropout=dropout)

        # ---- Soft fusion logits (one set per path) ----
        self.attn_fusion_logits = nn.Parameter(torch.zeros(3))
        self.mlp_fusion_logits  = nn.Parameter(torch.zeros(3))

        # ---- LayerScale γ ----
        ls = float(layer_scale_init)
        self.gamma_attn = nn.Parameter(ls * torch.ones(1, self.channel_dim, 1, 1))
        self.gamma_mlp  = nn.Parameter(ls * torch.ones(1, self.channel_dim, 1, 1))

        # ---- Dimension-Specific Stochastic Depth ----
        # Attention path: one DropPath per sub-branch (C / K / T).
        self.dp_c_attn = DropPath(drop_path_c)
        self.dp_k_attn = DropPath(drop_path_k)
        self.dp_t_attn = DropPath(drop_path_t)
        # MLP path: a single DropPath on the fused mlp branch output.
        self.dp_mlp    = DropPath(drop_path_mlp)

    # -------------------------------------------------------------------------
    # Branch computations
    # -------------------------------------------------------------------------

    def _attn_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-axis attention branch with per-axis DropPath (DSS)."""
        # C-view: norm along C, attend along K and along T (shared params).
        x_c = self.norm_c_attn(x)
        c_out = 0.5 * (
            self.attn_for_c(x_c, attend_dim=2, embed_axis=1)
            + self.attn_for_c(x_c, attend_dim=3, embed_axis=1)
        )
        c_out = self.dp_c_attn(c_out)

        # K-view
        x_k = self.norm_k_attn(x)
        k_out = 0.5 * (
            self.attn_for_k(x_k, attend_dim=1, embed_axis=2)
            + self.attn_for_k(x_k, attend_dim=3, embed_axis=2)
        )
        k_out = self.dp_k_attn(k_out)

        # T-view
        x_t = self.norm_t_attn(x)
        t_out = 0.5 * (
            self.attn_for_t(x_t, attend_dim=1, embed_axis=3)
            + self.attn_for_t(x_t, attend_dim=2, embed_axis=3)
        )
        t_out = self.dp_t_attn(t_out)

        w = torch.softmax(self.attn_fusion_logits, dim=0)
        return w[0] * c_out + w[1] * k_out + w[2] * t_out

    def _mlp_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Axis-wise MLP branch, single DropPath on the fused output."""
        x_c = self.norm_c_mlp(x)
        c_out = self.channel_mlp(x_c, dim=1)

        x_k = self.norm_k_mlp(x)
        k_out = self.k_mlp(x_k, dim=2)

        x_t = self.norm_t_mlp(x)
        t_out = self.t_mlp(x_t, dim=3)

        w = torch.softmax(self.mlp_fusion_logits, dim=0)
        fused = w[0] * c_out + w[1] * k_out + w[2] * t_out
        return self.dp_mlp(fused)

    # -------------------------------------------------------------------------
    # Forward: dual residual
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma_attn * self._attn_branch(x)
        x = x + self.gamma_mlp  * self._mlp_branch(x)
        return x


class TriAxisEncoder(nn.Module):
    """
    Stack of TriAxisMixerBlocks.

    Supports a per-layer DropPath schedule via `drop_path_schedule`:
        "linear":   per-layer rate scales linearly from 0 to the supplied
                    drop_path_{c,k,t,mlp} value (final layer gets the full rate).
        "constant": all layers use the same supplied rates.
    """

    def __init__(
        self,
        channel_dim: int,
        k_dim: int,
        t_dim: int,
        n_layers: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-2,
        drop_path_c: float = 0.0,
        drop_path_k: float = 0.0,
        drop_path_t: float = 0.0,
        drop_path_mlp: float = 0.0,
        drop_path_schedule: str = "linear",
    ):
        super().__init__()

        n_layers = max(1, int(n_layers))

        if drop_path_schedule == "linear" and n_layers > 1:
            rates = []
            for i in range(n_layers):
                s = i / (n_layers - 1)
                rates.append((
                    drop_path_c   * s,
                    drop_path_k   * s,
                    drop_path_t   * s,
                    drop_path_mlp * s,
                ))
        else:
            rates = [
                (drop_path_c, drop_path_k, drop_path_t, drop_path_mlp)
            ] * n_layers

        self.layers = nn.ModuleList([
            TriAxisMixerBlock(
                channel_dim=channel_dim,
                k_dim=k_dim,
                t_dim=t_dim,
                n_heads=n_heads,
                dropout=dropout,
                layer_scale_init=layer_scale_init,
                drop_path_c=rates[i][0],
                drop_path_k=rates[i][1],
                drop_path_t=rates[i][2],
                drop_path_mlp=rates[i][3],
            )
            for i in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False,
    ):
        """
        Args:
            x: input [B, D, K, T]
            return_all_layers: if True, return a list of per-layer outputs
                (length == n_layers) instead of only the final output.
                Used by MultiLevelTriAxisReadout.
        """
        if not return_all_layers:
            for layer in self.layers:
                x = layer(x)
            return x

        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


# =============================================================================
# Top-level model: TriDim (BasisMixer implementation class)
# =============================================================================

class BasisMixer(nn.Module):
    """
    TriDim model (BasisMixer is the implementation class).

    Pipeline:

        x_enc [B, L, C]
          -> transpose                                 [B, C, L]
          -> augmentation (training only)               [B, C, L]
          -> InstanceTimeNorm                           [B, C, L]
          -> ChannelAdapter (optional)                  [B, C', L]
          -> patchify                                   [B, C', K, T]
          -> channel_basis  (1x1 conv: C' -> D)         [B, D, K, T]
          -> k_basis        (Linear: K -> K')           [B, D, K', T]
          -> t_basis        (ConvT: T -> T')            [B, D, K', T']
          -> TriAxisEncoder (N TriDim blocks)              [B, D, K', T']
          -> Readout:
                 TriAxisAttentionPoolingHead (default)
                 OR MultiLevelTriAxisReadout    [B, num_class]

    Config fields:
        # Cross-subject frontend
        use_input_norm           bool   default True

        # Dimension-Specific Stochastic Depth
        drop_path_c              float  default 0.25
        drop_path_k              float  default 0.05
        drop_path_t              float  default 0.15
        drop_path_mlp            float  default 0.05
        drop_path_schedule       str    default "linear"

        # Multi-level readout
        use_multi_level_readout  bool   default False
            Replaces the single-layer TriAxisAttentionPoolingHead with
            MultiLevelTriAxisReadout, which pools every encoder layer's
            output independently and fuses them via learnable softmax
            weights before the final classifier.

    Setting use_input_norm=False, use_multi_level_readout=False, and all
    drop_path_*=0 disables these optional normalization, readout and stochastic-depth features.
    """

    def __init__(self, configs):
        super().__init__()

        # -------------------------
        # Basic configuration
        # -------------------------
        self.seq_len = int(_get_config(configs, "seq_len", 200))
        self.in_channels = int(_get_config(configs, "enc_in", 64))
        self.num_class = int(_get_config(configs, "num_class", 0))
        self.output_mode = str(_get_config(configs, "output_mode", "classification")).lower()

        self.patch_len = max(1, int(_get_config(configs, "patch_len", 16)))
        stride_cfg = _get_config(configs, "patch_stride", None)
        self.patch_stride = self.patch_len if stride_cfg is None else max(1, int(stride_cfg))

        self.use_channel_adapter = bool(_get_config(configs, "use_channel_adapter", False))
        self.use_channel_prior = bool(_get_config(configs, "use_channel_prior", True))
        self.canonical_channels = int(_get_config(configs, "canonical_channels", 64))

        self.channel_basis_dim = int(
            _get_config(configs, "channel_basis_dim", min(self.canonical_channels, 64))
        )
        self.k_basis_dim = int(
            _get_config(configs, "k_basis_dim", min(self.patch_len, 16))
        )
        self.t_basis_dim = int(
            _get_config(configs, "t_basis_dim", 16)
        )

        self.n_layers = max(1, int(_get_config(configs, "t_layer", 3)))
        self.n_heads = max(1, int(_get_config(configs, "n_heads", 4)))
        self.dropout = float(_get_config(configs, "dropout", 0.1))

        self.layer_scale_init = float(_get_config(configs, "layer_scale_init", 1e-2))

        self.patch_embed_dim = int(
            _get_config(configs, "patch_embed_dim", self.channel_basis_dim)
        )

        # -------------------------
        # cross-subject frontend flags
        # -------------------------
        self.use_input_norm = bool(_get_config(configs, "use_input_norm", True))

        # -------------------------
        # Dimension-Specific Stochastic Depth
        # -------------------------
        self.drop_path_c = float(_get_config(configs, "drop_path_c", 0.25))
        self.drop_path_k = float(_get_config(configs, "drop_path_k", 0.05))
        self.drop_path_t = float(_get_config(configs, "drop_path_t", 0.15))
        self.drop_path_mlp = float(_get_config(configs, "drop_path_mlp", 0.05))
        self.drop_path_schedule = str(_get_config(configs, "drop_path_schedule", "linear"))

        # -------------------------
        # multi-level readout
        # -------------------------
        self.use_multi_level_readout = bool(
            _get_config(configs, "use_multi_level_readout", False)
        )

        # -------------------------
        # Optional channel adapter
        # -------------------------
        canonical_coords = _get_config(configs, "canonical_channel_coords", None)
        canonical_names = _get_config(configs, "canonical_channel_names", None)
        canonical_coord_path = _get_config(configs, "canonical_channel_coord_path", None)

        if canonical_coord_path is not None:
            canonical_coords, canonical_names = load_channel_coordinate_csv(
                str(canonical_coord_path),
                expected_channels=self.canonical_channels,
                channel_order=canonical_names,
            )
        elif canonical_coords is not None:
            canonical_coords = torch.as_tensor(canonical_coords, dtype=torch.float32)

        if canonical_coords is None:
            canonical_coords = fibonacci_sphere(self.canonical_channels)

        self.channel_adapter = None
        active_channels = self.in_channels

        if self.use_channel_adapter:
            self.channel_adapter = ChannelAdapter(
                canonical_channels=self.canonical_channels,
                use_prior=self.use_channel_prior,
                sigma=float(_get_config(configs, "channel_adapter_sigma", 0.35)),
                residual_hidden_dim=int(_get_config(configs, "channel_adapter_hidden", 32)),
                residual_scale_init=float(
                    _get_config(configs, "channel_adapter_residual_scale", 0.10)
                ),
                canonical_coords=canonical_coords,
                canonical_channel_names=canonical_names,
            )
            active_channels = self.canonical_channels

        # -------------------------
        # input normalisation
        # -------------------------
        self.input_norm = (
            InstanceTimeNorm() if self.use_input_norm else nn.Identity()
        )

        # -------------------------
        # Axis projections
        # -------------------------
        self.channel_basis = LinearChannelProjection(
            active_channels, self.channel_basis_dim,
        )

        self.k_basis = LinearKProjection(
            self.patch_len, self.k_basis_dim,
        )

        self.t_basis = ConvTProjection(
            out_t_dim=self.t_basis_dim,
            kernel_size=int(_get_config(configs, "t_conv_kernel", 5)),
            dropout=self.dropout,
        )

        # -------------------------
        # Augmentations
        # -------------------------
        aug_specs = [
            s.strip()
            for s in str(_get_config(configs, "augmentations", "none")).split(",")
            if s.strip()
        ]
        if not aug_specs:
            aug_specs = ["none"]
        self.augmentations = nn.ModuleList(
            [get_augmentation(spec) for spec in aug_specs]
        )

        # -------------------------
        # Tri-axis encoder (TriDim block with DSS)
        # -------------------------
        self.encoder = TriAxisEncoder(
            self.channel_basis_dim,
            self.k_basis_dim,
            self.t_basis_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            layer_scale_init=self.layer_scale_init,
            drop_path_c=self.drop_path_c,
            drop_path_k=self.drop_path_k,
            drop_path_t=self.drop_path_t,
            drop_path_mlp=self.drop_path_mlp,
            drop_path_schedule=self.drop_path_schedule,
        )

        # -------------------------
        # Readout: single-layer or multi-level tri-axis pooling
        # -------------------------
        self.readout = None
        if self.num_class > 0:
            if self.use_multi_level_readout:
                self.readout = MultiLevelTriAxisReadout(
                    n_layers=self.n_layers,
                    channel_dim=self.channel_basis_dim,
                    k_dim=self.k_basis_dim,
                    t_dim=self.t_basis_dim,
                    embed_dim=self.patch_embed_dim,
                    num_class=self.num_class,
                    dropout=self.dropout,
                )
            else:
                self.readout = TriAxisAttentionPoolingHead(
                    channel_dim=self.channel_basis_dim,
                    k_dim=self.k_basis_dim,
                    t_dim=self.t_basis_dim,
                    embed_dim=self.patch_embed_dim,
                    num_class=self.num_class,
                    dropout=self.dropout,
                )

    # -------------------------------------------------------------------------
    # Patchify
    # -------------------------------------------------------------------------

    def _calc_num_patches(self, seq_len: int) -> int:
        if seq_len <= self.patch_len:
            return 1
        return int(math.ceil((seq_len - self.patch_len) / self.patch_stride)) + 1

    def _patchify(
        self,
        x_3d: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, t_len = x_3d.shape

        if t_len < self.patch_len:
            pad_len = self.patch_len - t_len
            x_3d = F.pad(x_3d, (0, pad_len))
            if seq_mask is not None:
                seq_mask = F.pad(seq_mask, (0, pad_len), value=False)
            t_len = self.patch_len

        n_patch = self._calc_num_patches(t_len)
        target_len = (n_patch - 1) * self.patch_stride + self.patch_len

        if t_len < target_len:
            pad_len = target_len - t_len
            x_3d = F.pad(x_3d, (0, pad_len))
            if seq_mask is not None:
                seq_mask = F.pad(seq_mask, (0, pad_len), value=False)

        x_patch = x_3d.unfold(
            dimension=2, size=self.patch_len, step=self.patch_stride,
        )
        x_4d = x_patch.permute(0, 1, 3, 2).contiguous()

        if seq_mask is not None:
            patch_mask = seq_mask.unfold(
                dimension=1, size=self.patch_len, step=self.patch_stride,
            ).any(dim=-1)
        else:
            patch_mask = torch.ones(
                x_4d.size(0), x_4d.size(-1),
                device=x_4d.device, dtype=torch.bool,
            )

        return x_4d, patch_mask

    def _make_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2).transpose(1, 2).contiguous()

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        x_enc: torch.Tensor,
        channel_coords: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        return_patch_embeddings: Optional[bool] = None,
    ):
        # [B, L, C] -> [B, C, L]
        x = x_enc.transpose(1, 2)

        # --- AUG FIRST (so input_norm absorbs aug-induced perturbations) ---
        if self.training and len(self.augmentations) > 0:
            aug = self.augmentations[random.randint(0, len(self.augmentations) - 1)]
            x = aug(x)

        # --- InstanceTimeNorm (raw EEG, removes subject baseline) ---
        x = self.input_norm(x)

        # --- Optional channel adapter (montage harmonisation) ---
        if self.channel_adapter is not None:
            x = self.channel_adapter(
                x, channel_coords=channel_coords, channel_mask=channel_mask,
            )

        # --- Patchify to [B, C, K, T] ---
        x, patch_mask = self._patchify(x, seq_mask=seq_mask)

        # --- Axis projections ---
        x = self.channel_basis(x)   # [B, C', K,  T ]
        x = self.k_basis(x)         # [B, C', K', T ]
        x = self.t_basis(x)         # [B, C', K', T']

        # --- Tri-axis encoder (TriDim block with DSS) ---
        # if multi-level readout is enabled, ask encoder for per-layer
        # outputs. Otherwise return only the final layer.
        if self.use_multi_level_readout and self.readout is not None:
            layer_outputs = self.encoder(x, return_all_layers=True)
            x = layer_outputs[-1]    # final layer for patch_embeddings / mode
        else:
            x = self.encoder(x)
            layer_outputs = None

        mode = self.output_mode
        if return_patch_embeddings is not None:
            mode = "patch" if return_patch_embeddings else "classification"

        if mode == "patch":
            return self._make_patch_embeddings(x)

        if self.readout is None:
            patch_embeddings = self._make_patch_embeddings(x)
            if mode == "both":
                return {
                    "patch_embeddings": patch_embeddings,
                    "patch_mask": patch_mask,
                    "logits": None,
                }
            return patch_embeddings

        # dispatch to the right readout signature.
        if isinstance(self.readout, MultiLevelTriAxisReadout):
            logits = self.readout(layer_outputs)
        else:
            logits = self.readout(x)

        if mode == "both":
            return {
                "patch_embeddings": self._make_patch_embeddings(x),
                "patch_mask": patch_mask,
                "logits": logits,
            }

        return logits


Model = BasisMixer


# =============================================================================
# Ablation config presets
# =============================================================================
#
# Use these as the rows of your paper's Table 4 (Ablation Study).
# Each config builds on top of the previous one — only one component changes
# at a time, so the marginal effect of each is isolated.
#
# Suggested experiment order:
#   1. TriDim baseline                  — encoder without optional features
#   2. + InstanceTimeNorm           — cross-subject baseline normalisation
#   3. + Temporal stem              — frequency-band prior
#   4. + DSS DropPath               — dimension-specific regularisation
# =============================================================================

class _BaseConfig:
    """Shared training-side config defaults."""
    seq_len = 200
    enc_in = 62
    num_class = 5
    patch_len = 25
    patch_stride = 25
    channel_basis_dim = 48
    k_basis_dim = 16
    t_basis_dim = 12
    patch_embed_dim = 48
    n_heads = 4
    t_layer = 2
    dropout = 0.1
    output_mode = "both"
    use_channel_adapter = False
    augmentations = "none"
    layer_scale_init = 1e-2


class cfg_v5_baseline(_BaseConfig):
    """Pure TriDim — no input norm, no DSS, no multi-level readout."""
    use_input_norm = False
    use_multi_level_readout = False
    drop_path_c = 0.0
    drop_path_k = 0.0
    drop_path_t = 0.0
    drop_path_mlp = 0.0


class cfg_v11_plus_inorm(_BaseConfig):
    """TriDim + InstanceTimeNorm at the input."""
    use_input_norm = True
    use_multi_level_readout = False
    drop_path_c = 0.0
    drop_path_k = 0.0
    drop_path_t = 0.0
    drop_path_mlp = 0.0


class cfg_v11_inorm_dss(_BaseConfig):
    """TriDim + InstanceTimeNorm + DSS DropPath (no multi-level readout)."""
    use_input_norm = True
    use_multi_level_readout = False
    drop_path_c = 0.25
    drop_path_k = 0.05
    drop_path_t = 0.15
    drop_path_mlp = 0.05
    drop_path_schedule = "linear"


class cfg_v11_2_full(_BaseConfig):
    """Full TriDim: TriDim + InstanceTimeNorm + DSS + multi-level readout.

    This is the post-stem-removal model. Novelty is concentrated in the
    tri-axis backbone, the dimension-specific stochastic depth, and the
    multi-level readout — no EEGNet-style temporal/spatial stem.
    """
    use_input_norm = True
    use_multi_level_readout = True
    drop_path_c = 0.25
    drop_path_k = 0.05
    drop_path_t = 0.15
    drop_path_mlp = 0.05
    drop_path_schedule = "linear"


class cfg_v11_2_full_constant_dss(_BaseConfig):
    """Full TriDim with constant DSS schedule + stronger C drop_path.

    Recommended starting point: constant schedule keeps shallow layers
    regularised (linear schedule drives early-layer rates toward 0, which
    is wrong for shallow nets).
    """
    use_input_norm = True
    use_multi_level_readout = True
    drop_path_c = 0.35
    drop_path_k = 0.05
    drop_path_t = 0.20
    drop_path_mlp = 0.05
    drop_path_schedule = "constant"


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":

    torch.set_num_threads(1)

    print("=" * 70)
    print("TriDim smoke test — ablation comparison")
    print("=" * 70)

    test_configs = [
        ("TriDim baseline",                cfg_v5_baseline),
        ("TriDim + InstanceTimeNorm",      cfg_v11_plus_inorm),
        ("TriDim + InstanceNorm + DSS",    cfg_v11_inorm_dss),
        ("Full TriDim (linear DSS)",    cfg_v11_2_full),
        ("Full TriDim (constant DSS)",  cfg_v11_2_full_constant_dss),
    ]

    x = torch.randn(2, _BaseConfig.seq_len, _BaseConfig.enc_in)
    seq_mask = torch.ones(2, _BaseConfig.seq_len, dtype=torch.bool)

    for name, cfg_cls in test_configs:
        print()
        print(f"--- {name} ---")
        cfg = cfg_cls()
        model = Model(cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trainable params: {n_params:,}")

        model.eval()
        out = model(x, seq_mask=seq_mask)
        print(f"  patch_embeddings: {tuple(out['patch_embeddings'].shape)}")
        print(f"  logits:           {tuple(out['logits'].shape)}")

        # Backward sanity
        model.train()
        out = model(x, seq_mask=seq_mask)
        loss = out["logits"].sum()
        loss.backward()
        print(f"  backward OK; loss = {float(loss.detach()):.4f}")

    # ----- Detailed inspection of Full TriDim -----
    print()
    print("=" * 70)
    print("Full TriDim (constant DSS) — detailed inspection")
    print("=" * 70)
    cfg = cfg_v11_2_full_constant_dss()
    model = Model(cfg)

    print()
    print(f"input_norm:    {type(model.input_norm).__name__}")
    print(f"(temporal stem removed in this version)")
    print(f"channel_basis: {type(model.channel_basis).__name__}")
    print(f"readout type:  {type(model.readout).__name__}")
    if isinstance(model.readout, MultiLevelTriAxisReadout):
        print(f"  n_layers in readout: {model.readout.n_layers}")
        print(
            f"  initial layer fusion softmax: "
            f"{torch.softmax(model.readout.layer_fusion_logits, dim=0).detach().cpu().tolist()}"
        )

    print()
    print("DSS DropPath schedule (per layer):")
    for i, layer in enumerate(model.encoder.layers):
        print(
            f"  layer {i}:  dp_c={layer.dp_c_attn.drop_prob:.4f}  "
            f"dp_k={layer.dp_k_attn.drop_prob:.4f}  "
            f"dp_t={layer.dp_t_attn.drop_prob:.4f}  "
            f"dp_mlp={layer.dp_mlp.drop_prob:.4f}"
        )

    print()
    blk = model.encoder.layers[0]
    print("First-block fusion logits (initialised at zero -> uniform softmax):")
    print(
        f"  attn_fusion_softmax: "
        f"{torch.softmax(blk.attn_fusion_logits, dim=0).detach().cpu().tolist()}"
    )
    print(
        f"  mlp_fusion_softmax:  "
        f"{torch.softmax(blk.mlp_fusion_logits, dim=0).detach().cpu().tolist()}"
    )
    print(f"  gamma_attn mean: {float(blk.gamma_attn.mean().detach()):.4f}")
    print(f"  gamma_mlp  mean: {float(blk.gamma_mlp.mean().detach()):.4f}")
