"""
EEG Mixer V11.2 - Criss-Cross block variant (S-fixed-as-embedding)
=============================================================================

Reviewer-response ablation for "Beyond Flattened Tokens".  The Full TriDim
block rotates the axis role: each of C / K(=S) / T(=L) takes a turn as the
embedding dimension while attention runs over the other two axes (three
views, softmax-fused).  This criss-cross variant FIXES the S axis (K, the
local spectral axis) as the embedding dimension and computes only that
view's two attention ops:

    Attn_{c|s}: attention over the C axis, S as the embedding dimension
    Attn_{l|s}: attention over the T (global patch) axis, S as embedding

The C-as-embedding and L-as-embedding views, together with the attention
softmax fusion across views, are removed; the single remaining view feeds
the residual directly.  Everything else is byte-identical to Full:
axis-specific FFN path (channel / k / t MLPs with softmax fusion and a
single DropPath), per-axis RMSNorm on the FFN path, per-channel LayerScale
gamma on both residual paths, dual-residual skeleton, DSS drop-path wiring,
TriAxisEncoder, projections (channel / k / t basis), InstanceTimeNorm, and
both readout heads (TriAxisAttentionPoolingHead / MultiLevelTriAxisReadout).

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
# Criss-Cross Mixer Block - S fixed as the embedding dimension
# =============================================================================

class CrissCrossMixerBlock(nn.Module):
    """
    Criss-cross block: the S (K / local-spectral) axis is FIXED as the
    embedding dimension.  Only the S-as-embedding view's two attention ops
    are computed (Attn_{c|s} + Attn_{l|s}); the attention softmax fusion is
    dropped and the single view feeds the residual directly.

    The FFN path (three axis-specific MLPs with softmax fusion and a single
    DropPath) and both LayerScale gammas are identical to Full.
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
        # drop_path_c / drop_path_t / axis_branch_drop_* are accepted for
        # signature compatibility with Full but are unused: the criss-cross
        # block has a single (S) attention sub-branch.

        # ---- RMSNorm: S-axis norm for attention; per-axis norms for MLP ----
        self.norm_k_attn = AxisRMSNorm(self.k_dim,       axis=2)
        self.norm_c_mlp = AxisRMSNorm(self.channel_dim, axis=1)
        self.norm_k_mlp = AxisRMSNorm(self.k_dim,       axis=2)
        self.norm_t_mlp = AxisRMSNorm(self.t_dim,       axis=3)

        # ---- Criss-cross attention: S stays the embedding axis ----
        self.attn_for_k = AxisAttention(self.k_dim, n_heads=n_heads, dropout=dropout)

        # ---- Axis-wise MLPs (identical to Full) ----
        self.channel_mlp = AxisMLP(self.channel_dim, dropout=dropout)
        self.k_mlp       = AxisMLP(self.k_dim,       dropout=dropout)
        self.t_mlp       = AxisMLP(self.t_dim,       dropout=dropout)

        # ---- Soft fusion logits (MLP path only, identical to Full) ----
        self.mlp_fusion_logits  = nn.Parameter(torch.zeros(3))

        # ---- LayerScale gamma (identical to Full) ----
        ls = float(layer_scale_init)
        self.gamma_attn = nn.Parameter(ls * torch.ones(1, self.channel_dim, 1, 1))
        self.gamma_mlp  = nn.Parameter(ls * torch.ones(1, self.channel_dim, 1, 1))

        # ---- DropPath: single S attention sub-branch + single MLP path ----
        self.dp_k_attn = DropPath(drop_path_k)
        self.dp_mlp    = DropPath(drop_path_mlp)

    # ---------------------------------------------------------------------
    # Branch computations
    # ---------------------------------------------------------------------

    def _attn_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Criss-cross attention: S-as-embedding view only, direct connect."""
        # S-view: norm along S(=K); attend over C and over T (shared params).
        x_k = self.norm_k_attn(x)
        k_out = 0.5 * (
            self.attn_for_k(x_k, attend_dim=1, embed_axis=2)   # Attn_{c|s}
            + self.attn_for_k(x_k, attend_dim=3, embed_axis=2) # Attn_{l|s}
        )
        return self.dp_k_attn(k_out)

    def _mlp_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Axis-wise MLP branch, identical to Full."""
        x_c = self.norm_c_mlp(x)
        c_out = self.channel_mlp(x_c, dim=1)

        x_k = self.norm_k_mlp(x)
        k_out = self.k_mlp(x_k, dim=2)

        x_t = self.norm_t_mlp(x)
        t_out = self.t_mlp(x_t, dim=3)

        w = torch.softmax(self.mlp_fusion_logits, dim=0)
        fused = w[0] * c_out + w[1] * k_out + w[2] * t_out
        return self.dp_mlp(fused)

    # ---------------------------------------------------------------------
    # Forward: dual residual (identical skeleton to Full)
    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma_attn * self._attn_branch(x)
        x = x + self.gamma_mlp  * self._mlp_branch(x)
        return x


class TriAxisEncoder(nn.Module):
    """
    Stack of CrissCrossMixerBlocks.

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
            CrissCrossMixerBlock(
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
        # Tri-axis encoder (V11 block with DSS)
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
            axis_branch_drop_prob=self.axis_branch_drop_prob,
            axis_branch_drop_mode=self.axis_branch_drop_mode,
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
