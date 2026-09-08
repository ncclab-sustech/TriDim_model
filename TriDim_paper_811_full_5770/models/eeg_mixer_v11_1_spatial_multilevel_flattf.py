"""
EEG Mixer V11.2 〞 V5 backbone + InstanceTimeNorm + DSS + Multi-Level Readout
=============================================================================

This version removes the multi-scale temporal stem (and its spatial-mix
sub-module) that earlier V11.1 included. The decision: the stem was too
close to EEGNet's temporal+spatial conv and diluted the contribution story.
With the stem gone, the architecture's novelty is concentrated entirely in
the tri-axis backbone, the dimension-specific stochastic depth, and the
multi-level readout 〞 none of which overlap with EEGNet.

Components on top of the V5 backbone (all OUTSIDE the backbone):

    (1) InstanceTimeNorm at the front
        Per-sample per-channel time-axis normalisation; removes subject
        DC offset and amplitude scale. The single most important component
        for cross-subject generalisation. 0 parameters.
        Toggle: `use_input_norm = True/False`.

    (2) Dimension-Specific Stochastic Depth (DSS)
        Each attention sub-branch (C / K / T) gets its own DropPath rate;
        the MLP path gets one rate. C axis (most subject-sensitive) uses a
        high rate, K axis (most subject-stable) a low rate, T axis a medium
        rate. No reported prior work sets per-axis stochastic-depth rates.
        Config: drop_path_c / drop_path_k / drop_path_t / drop_path_mlp.

    (3) Multi-level tri-axis readout
        Instead of using only the encoder's final-layer output, every
        layer's [B, D, K, T] is independently pooled (sharing the
        TriAxisAttentionPoolingHead design from V5) and the per-layer
        pooled representations are fused with a learnable softmax weight,
        then passed through a single classifier head. Lightweight analogue
        of EEG-Deformer's Dense Information Purification (DIP) 〞 improves
        gradient flow to shallow layers, gives a multi-scale view.
        Toggle: `use_multi_level_readout = True/False`.

REMOVED relative to V11.1:
    * TemporalMultiScaleStem  (multi-scale depthwise temporal conv)
    * spatial_mix             (the 1x1 channel-mixing inside the stem)
    The entry point reverts to V5's plain 1x1 channel projection
    (LinearChannelProjection) directly before patchify.

V5 backbone (untouched):
    * parallel tri-axis with softmax fusion (separate logits for attn / mlp)
    * per-axis RMSNorm x 6 per block
    * shared cross-axis attention (3 modules, each invoked twice)
    * per-channel LayerScale gamma on attn and mlp residual paths


=============================================================================
VARIANT flattf (Flattened Transformer) 〞 block-level ablation
=============================================================================

This variant replaces the TriAxisEncoder with a FlattenedEncoder: the
[B, C, K, T] tensor is reshaped into a plain token sequence
[B, K*T, C] (one token per (k, t) position, feature dim = channel_dim)
and processed by n_layers standard prenorm Transformer encoder layers
(MHSA over the flattened tokens + two-layer expansion=2 MLP, with the
same n_heads / dropout / LayerScale init=1e-2 as the base, and a
DropPath linear schedule whose final-layer rate equals drop_path_mlp).
The sequence is then reshaped back to [B, C, K, T].

Purpose: verify whether the explicit tri-axis structure of the TriDim
block is actually superior to an ordinary flattened-token standard
Transformer. Everything outside the encoder (patchify, basis
projections, InstanceTimeNorm, readout, classifier head) is kept
byte-identical to the base model.

The FlattenedEncoder supports the same return_all_layers interface as
TriAxisEncoder, so MultiLevelTriAxisReadout consumes per-layer outputs
of shape [B, C, K, T] exactly as in the base model.

"""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



def _get_config(configs, name: str, default):
    return getattr(configs, name, default)


# =============================================================================
# NEW: Cross-subject normalisation at the input
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
# NEW: DropPath (per-sample stochastic depth) 〞 needed for DSS
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
# Axis projections  (unchanged from V4 / V5)
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

    The shared 1-to-1 Conv1d is intentionally retained because it is part of
    the reported architecture and published checkpoint parameterization.
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
# Tri-axis attention pooling readout  (unchanged from V4 / V5)
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
    skipped 〞 the module returns the fused embed_dim vector instead. This
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
    V11.1 multi-level tri-axis readout (DIP-inspired).

    Given the per-layer outputs from the encoder
        [layer_0_out, layer_1_out, ..., layer_{N-1}_out]
    each shaped [B, D, K, T], this readout:

        1. Runs an independent TriAxisAttentionPoolingHead (in feature mode)
           on each layer's output, producing a per-layer embed_dim vector.
        2. Fuses the per-layer vectors with a learnable softmax weight
           (n_layers entries, initialised to zero ↙ uniform after softmax).
        3. Applies a single classifier head to the fused vector.

    Compared to using only the final layer:
      * Improves gradient flow to shallow layers (extra supervision signal).
      * Gives the classifier a multi-scale view of the encoder hierarchy.
      * Cheap: each per-layer pool reuses the well-tested
        TriAxisAttentionPoolingHead design from V5.

    Compared to EEG-Deformer's Dense Information Purification:
      * Lightweight: no dense skip connections through the backbone, just
        late-stage pooling.
      * Fully decoupled from the backbone, which keeps the V5 tri-axis
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

        # One pooling head per encoder layer, sharing the V5 TriAxis design
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
# Building blocks for the encoder  (unchanged from V5)
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
# V11 Tri-axis Mixer Block 〞 V5 backbone + DSS DropPath per sub-branch
# =============================================================================

class TriAxisMixerBlock(nn.Module):
    """
    V11 PreNorm + dual-residual + shared cross-axis attention
        + per-channel LayerScale + Dimension-Specific Stochastic Depth (DSS).

    Structure:

        # Attention path: 3 sub-branches in parallel, each with its own DropPath
        c_out = dp_c(0.5 * (attn_for_c(norm_c, attend=K) + attn_for_c(norm_c, attend=T)))
        k_out = dp_k(0.5 * (attn_for_k(norm_k, attend=C) + attn_for_k(norm_k, attend=T)))
        t_out = dp_t(0.5 * (attn_for_t(norm_t, attend=C) + attn_for_t(norm_t, attend=K)))
        attn_branch = w_attn ﹞-fused (c_out, k_out, t_out)
        x = x + 污_attn × attn_branch

        # MLP path: 3 sub-branches in parallel, single DropPath at the end
        c_out = channel_mlp(norm_c_mlp, dim=1)
        k_out = k_mlp(norm_k_mlp, dim=2)
        t_out = t_mlp(norm_t_mlp, dim=3)
        mlp_branch = dp_mlp(w_mlp ﹞-fused (c_out, k_out, t_out))
        x = x + 污_mlp × mlp_branch

    Differences from V5:
        * Each attention sub-branch is gated by its own DropPath rate (DSS).
          Default: drop_path_c=0.25, drop_path_k=0.05, drop_path_t=0.15.
        * The MLP path is gated by a single DropPath rate (drop_path_mlp).
        * All four rates are configurable per block (used by the encoder
          to implement a linear DropPath schedule across layers).

    Backwards-compatible: setting all drop_path_* to 0 reproduces V5 exactly.
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
        axis_branch_drop_prob: float = 0.0,
        axis_branch_drop_mode: str = "random_one",
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

        # ---- LayerScale 污 ----
        ls = float(layer_scale_init)
        self.gamma_attn = nn.Parameter(ls * torch.ones(1, self.channel_dim, 1, 1))
        self.gamma_mlp  = nn.Parameter(ls * torch.ones(1, self.channel_dim, 1, 1))

        # ---- NEW: Dimension-Specific Stochastic Depth ----
        # Attention path: one DropPath per sub-branch (C / K / T).
        self.dp_c_attn = DropPath(drop_path_c)
        self.dp_k_attn = DropPath(drop_path_k)
        self.dp_t_attn = DropPath(drop_path_t)
        # MLP path: a single DropPath on the fused mlp branch output.
        self.dp_mlp    = DropPath(drop_path_mlp)

        # Optional training-only axis branch dropout. With random_one, one
        # attention axis branch is masked per batch and surviving branches are
        # scaled so the branch expectation is preserved. Default 0 is no-op.
        self.axis_branch_drop_prob = max(0.0, min(1.0, float(axis_branch_drop_prob)))
        self.axis_branch_drop_mode = str(axis_branch_drop_mode).lower()

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

        if self.training and self.axis_branch_drop_prob > 0.0:
            if self.axis_branch_drop_mode != "random_one":
                raise ValueError(f"Unsupported axis_branch_drop_mode: {self.axis_branch_drop_mode}")
            if torch.rand((), device=x.device) < self.axis_branch_drop_prob:
                drop_idx = int(torch.randint(0, 3, (), device=x.device).item())
                scale = 1.5
                if drop_idx == 0:
                    c_out = torch.zeros_like(c_out)
                    k_out = k_out * scale
                    t_out = t_out * scale
                elif drop_idx == 1:
                    k_out = torch.zeros_like(k_out)
                    c_out = c_out * scale
                    t_out = t_out * scale
                else:
                    t_out = torch.zeros_like(t_out)
                    c_out = c_out * scale
                    k_out = k_out * scale

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
        axis_branch_drop_prob: float = 0.0,
        axis_branch_drop_mode: str = "random_one",
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
                axis_branch_drop_prob=axis_branch_drop_prob,
                axis_branch_drop_mode=axis_branch_drop_mode,
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
# VARIANT flattf: Flattened Transformer block / encoder
# =============================================================================

class FlattenedMHSA(nn.Module):
    """Standard multi-head self-attention over a token sequence [B, N, D]."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class FlattenedMLP(nn.Module):
    """Two-layer token MLP with expansion=2 (mirrors AxisMLP structure)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlattenedTransformerBlock(nn.Module):
    """
    Standard prenorm Transformer encoder layer on flattened tokens.

    Structure (tokens x: [B, N, D], N = K*T, D = channel_dim):

        x = x + gamma_attn * dp(MHSA(LayerNorm(x)))
        x = x + gamma_mlp  * dp(MLP(LayerNorm(x)))

    LayerScale (init=1e-2, per-channel) and DropPath are applied on both
    residual paths, matching the base block's gating design.
    """

    def __init__(
        self,
        channel_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.channel_dim = int(channel_dim)

        self.norm_attn = nn.LayerNorm(self.channel_dim)
        self.norm_mlp = nn.LayerNorm(self.channel_dim)

        self.attn = FlattenedMHSA(self.channel_dim, n_heads=n_heads, dropout=dropout)
        self.mlp = FlattenedMLP(self.channel_dim, dropout=dropout)

        ls = float(layer_scale_init)
        self.gamma_attn = nn.Parameter(ls * torch.ones(self.channel_dim))
        self.gamma_mlp = nn.Parameter(ls * torch.ones(self.channel_dim))

        self.dp_attn = DropPath(drop_path)
        self.dp_mlp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma_attn * self.dp_attn(self.attn(self.norm_attn(x)))
        x = x + self.gamma_mlp * self.dp_mlp(self.mlp(self.norm_mlp(x)))
        return x


class FlattenedEncoder(nn.Module):
    """
    Stack of FlattenedTransformerBlocks over flattened (K*T) tokens.

    The [B, C, K, T] input is reshaped to a token sequence
    [B, K*T, C] (one token per (k, t) position, feature dim = C), run
    through the Transformer layers, and reshaped back to [B, C, K, T].

    Uses the same DropPath schedule rule as TriAxisEncoder ("linear"
    scales the rate linearly from 0 to the supplied value, final layer
    gets the full rate), with drop_path_mlp as the single target rate.
    Supports the same return_all_layers interface as TriAxisEncoder;
    every returned layer output has shape [B, C, K, T].
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
        drop_path: float = 0.0,
        drop_path_schedule: str = "linear",
    ):
        super().__init__()
        self.channel_dim = int(channel_dim)
        self.k_dim = int(k_dim)
        self.t_dim = int(t_dim)

        n_layers = max(1, int(n_layers))

        if drop_path_schedule == "linear" and n_layers > 1:
            rates = [drop_path * (i / (n_layers - 1)) for i in range(n_layers)]
        else:
            rates = [drop_path] * n_layers

        self.layers = nn.ModuleList([
            FlattenedTransformerBlock(
                channel_dim=channel_dim,
                n_heads=n_heads,
                dropout=dropout,
                layer_scale_init=layer_scale_init,
                drop_path=rates[i],
            )
            for i in range(n_layers)
        ])

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, c, k, t = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, k * t, c)

    def _to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        return (
            tokens.reshape(b, self.k_dim, self.t_dim, c)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False,
    ):
        """
        Args:
            x: input [B, D, K, T]
            return_all_layers: if True, return a list of per-layer outputs
                (length == n_layers, each [B, D, K, T]) instead of only
                the final output. Used by MultiLevelTriAxisReadout.
        """
        tokens = self._to_tokens(x)
        if not return_all_layers:
            for layer in self.layers:
                tokens = layer(tokens)
            return self._to_grid(tokens)

        outputs = []
        for layer in self.layers:
            tokens = layer(tokens)
            outputs.append(self._to_grid(tokens))
        return outputs


# =============================================================================
# Top-level model: BasisMixer (V11)
# =============================================================================

class BasisMixer(nn.Module):
    """
    V11.1 BasisMixer.

    Pipeline:

        x_enc [B, L, C]
          -> transpose                                 [B, C, L]
          -> InstanceTimeNorm                           [B, C, L]
          -> patchify                                   [B, C', K, T]
          -> channel_basis  (1x1 conv: C' -> D)         [B, D, K, T]
          -> k_basis        (Linear: K -> K')           [B, D, K', T]
          -> t_basis        (ConvT: T -> T')            [B, D, K', T']
          -> TriAxisEncoder (V5 block ℅ N)              [B, D, K', T']
          -> Readout:
                 TriAxisAttentionPoolingHead (default)
                 OR MultiLevelTriAxisReadout (V11.1)    [B, num_class]

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
    drop_path_*=0 reproduces V5.
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
        # Cross-subject frontend
        # -------------------------
        self.use_input_norm = bool(_get_config(configs, "use_input_norm", True))

        # -------------------------
        # Dimension-specific stochastic depth
        # -------------------------
        self.drop_path_c = float(_get_config(configs, "drop_path_c", 0.25))
        self.drop_path_k = float(_get_config(configs, "drop_path_k", 0.05))
        self.drop_path_t = float(_get_config(configs, "drop_path_t", 0.15))
        self.drop_path_mlp = float(_get_config(configs, "drop_path_mlp", 0.05))
        self.drop_path_schedule = str(_get_config(configs, "drop_path_schedule", "linear"))
        self.axis_branch_drop_prob = float(_get_config(configs, "axis_branch_drop_prob", 0.0))
        self.axis_branch_drop_mode = str(_get_config(configs, "axis_branch_drop_mode", "random_one"))

        # -------------------------
        # Multi-level readout
        # -------------------------
        self.use_multi_level_readout = bool(
            _get_config(configs, "use_multi_level_readout", False)
        )

        # -------------------------
        # Channel adapter removed (public release)
        # -------------------------
        active_channels = self.in_channels

        # -------------------------
        # NEW: input normalisation
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
        # Flattened Transformer encoder (VARIANT flattf)
        # -------------------------
        self.encoder = FlattenedEncoder(
            self.channel_basis_dim,
            self.k_basis_dim,
            self.t_basis_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            layer_scale_init=self.layer_scale_init,
            drop_path=self.drop_path_mlp,
            drop_path_schedule=self.drop_path_schedule,
        )

        # -------------------------
        # Readout: V5 single-layer head OR V11.1 multi-level head
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

        # --- InstanceTimeNorm (raw EEG, removes subject baseline) ---
        x = self.input_norm(x)

        # --- Patchify to [B, C, K, T] ---
        x, patch_mask = self._patchify(x, seq_mask=seq_mask)

        # --- Axis projections ---
        x = self.channel_basis(x)   # [B, C', K,  T ]
        x = self.k_basis(x)         # [B, C', K', T ]
        x = self.t_basis(x)         # [B, C', K', T']

        # --- Tri-axis encoder (V5 block with DSS) ---
        # If multi-level readout is enabled, request every encoder layer;
        # otherwise return only the final layer.
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

        # Dispatch according to the configured readout.
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
