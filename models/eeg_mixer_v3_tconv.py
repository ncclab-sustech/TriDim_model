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


def _get_config(configs, name: str, default):
    return getattr(configs, name, default)


def load_channel_coordinate_csv(csv_path: str, expected_channels: Optional[int] = None, channel_order: Optional[Sequence[str]] = None, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, Sequence[str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Channel coordinate CSV not found: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"channel", "x", "y", "z"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CSV {csv_path} must contain columns {sorted(required)}, missing {sorted(missing)}")
        rows = []
        for row in reader:
            rows.append((str(row["channel"]).strip(), [float(row["x"]), float(row["y"]), float(row["z"])]))
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
        raise ValueError(f"Expected {expected_channels} channels, got {coords.size(0)} from {csv_path}")
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


def pairwise_rbf_logits(target_coords: torch.Tensor, source_coords: torch.Tensor, sigma: float) -> torch.Tensor:
    rel = target_coords.unsqueeze(-2) - source_coords.unsqueeze(-3)
    dist2 = (rel ** 2).sum(dim=-1)
    sigma2 = max(float(sigma), 1e-6) ** 2
    return -dist2 / (2.0 * sigma2)


def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
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
        self.net = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, target_coords: torch.Tensor, source_coords: torch.Tensor) -> torch.Tensor:
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
                raise ValueError("Batch mismatch between target_coords and source_coords")
        rel = target_coords.unsqueeze(2) - source_coords.unsqueeze(1)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        return self.net(torch.cat([rel, dist], dim=-1)).squeeze(-1)


class ChannelAdapter(nn.Module):
    def __init__(self, canonical_channels: int = 64, use_prior: bool = True, sigma: float = 0.35, residual_hidden_dim: int = 32, residual_scale_init: float = 0.10, canonical_coords: Optional[torch.Tensor] = None, canonical_channel_names: Optional[Sequence[str]] = None):
        super().__init__()
        self.canonical_channels = int(canonical_channels)
        self.use_prior = bool(use_prior)
        self.sigma = float(sigma)
        self.residual_mapper = CoordinateResidualMapper(hidden_dim=residual_hidden_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        if canonical_coords is None:
            canonical_coords = fibonacci_sphere(self.canonical_channels)
        self.register_buffer("canonical_coords", canonical_coords.float(), persistent=True)
        self.canonical_channel_names = list(canonical_channel_names) if canonical_channel_names is not None else None

    def set_canonical_coords(self, coords: torch.Tensor, channel_names: Optional[Sequence[str]] = None) -> None:
        self.canonical_channels = int(coords.size(0))
        self.register_buffer("canonical_coords", coords.float(), persistent=True)
        self.canonical_channel_names = list(channel_names) if channel_names is not None else self.canonical_channel_names

    def _identity_map(self, cin: int, device, dtype) -> torch.Tensor:
        eye = torch.eye(cin, device=device, dtype=dtype)
        if cin == self.canonical_channels:
            return eye
        out = torch.zeros(self.canonical_channels, cin, device=device, dtype=dtype)
        copy_n = min(cin, self.canonical_channels)
        out[:copy_n, :copy_n] = eye[:copy_n, :copy_n]
        return out

    def _compute_weights(self, channel_coords: torch.Tensor, channel_mask: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if channel_coords.ndim == 2:
            channel_coords = channel_coords.unsqueeze(0)
        batch_size = channel_coords.size(0)
        target_coords = self.canonical_coords.to(device=channel_coords.device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        geom_logits = pairwise_rbf_logits(target_coords, channel_coords.to(dtype=dtype), sigma=self.sigma)
        learned_logits = self.residual_mapper(target_coords, channel_coords.to(dtype=dtype))
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

    def forward(self, x: torch.Tensor, channel_coords: Optional[torch.Tensor] = None, channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, cin, _ = x.shape
        if channel_coords is None:
            if cin != self.canonical_channels:
                raise ValueError("channel_coords is required when input channel count differs from canonical_channels")
            weights = self._identity_map(cin, x.device, x.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            weights = self._compute_weights(channel_coords, channel_mask, x.dtype)
            if weights.size(0) == 1 and batch_size > 1:
                weights = weights.expand(batch_size, -1, -1)
        return torch.einsum("boc,bct->bot", weights.to(dtype=x.dtype), x)


class LinearChannelProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LinearKProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.transpose(2, 3)
        out = self.proj(x_perm)
        return out.transpose(2, 3).contiguous()

class LinearTProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        return out.contiguous()

class PoolTProjection(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.pool = nn.AdaptiveAvgPool1d(self.out_dim)

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, channels, k_dim, t_dim = x.shape
        if patch_mask is None:
            if t_dim == self.out_dim:
                return x
            x_flat = x.reshape(batch_size, channels * k_dim, t_dim)
            return self.pool(x_flat).reshape(batch_size, channels, k_dim, self.out_dim)
        valid_lengths = patch_mask.sum(dim=1).clamp(min=1, max=t_dim).to(torch.long)
        if bool((valid_lengths == t_dim).all().item()) and t_dim == self.out_dim:
            return x
        out = torch.zeros(batch_size, channels, k_dim, self.out_dim, device=x.device, dtype=x.dtype)
        for b in range(batch_size):
            valid = int(valid_lengths[b].item())
            sample = x[b, :, :, :valid].reshape(1, channels * k_dim, valid)
            out[b] = self.pool(sample).reshape(channels, k_dim, self.out_dim)
        return out

class ConvTProjection(nn.Module):
    """
    Convolutional projection along the global temporal patch axis T.

    Input:
        x: [B, C, K, T_in]

    Output:
        y: [B, C, K, T_out]

    This replaces LinearTProjection(patch_num, t_basis_dim) with:
        learnable Conv1d along T + non-parametric AdaptiveAvgPool1d.
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
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(self.out_t_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, K, T_in]
        b, c, k, t = x.shape

        # Treat every (B, C, K) trajectory as a 1D sequence over T.
        y = x.reshape(b * c * k, 1, t)

        # Learnable temporal filtering.
        y = self.proj(y)

        # Fixed output length.
        y = self.pool(y)

        # Back to [B, C, K, T_out]
        y = y.reshape(b, c, k, self.out_t_dim)

        return y
    
# class AxisConvMLP(nn.Module):
#     def __init__(
#         self,
#         dim_size: int,
#         dropout: float = 0.0,
#         hidden: int = 16,
#         kernel_size: int = 5,
#         dilation: int = 1,
#     ):
#         super().__init__()
#         self.dim_size = int(dim_size)

#         padding = dilation * (kernel_size - 1) // 2

#         self.net = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=1,
#                 out_channels=hidden,
#                 kernel_size=kernel_size,
#                 padding=padding,
#                 dilation=dilation,
#             ),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Conv1d(
#                 in_channels=hidden,
#                 out_channels=1,
#                 kernel_size=1,
#             ),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
#         # Move target axis to the last dimension.
#         x_perm = x.transpose(dim, -1)
#         original_shape = x_perm.shape

#         # [*, T] -> [N, 1, T]
#         x_flat = x_perm.reshape(-1, self.dim_size).unsqueeze(1)

#         out = self.net(x_flat).squeeze(1)

#         # In case padding/dilation causes minor length mismatch.
#         if out.shape[-1] != self.dim_size:
#             out = out[..., : self.dim_size]

#         out = out.reshape(original_shape)
#         return out.transpose(dim, -1)
    

class AxisMLP(nn.Module):
    def __init__(self, dim_size: int, dropout: float = 0.0, expansion: int = 2):
        super().__init__()
        self.dim_size = int(dim_size)
        hidden = self.dim_size * int(expansion)
        self.net = nn.Sequential(nn.Linear(self.dim_size, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, self.dim_size), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        x_perm = x.transpose(dim, -1)
        original_shape = x_perm.shape
        out = self.net(x_perm.reshape(-1, self.dim_size)).reshape(original_shape)
        return out.transpose(dim, -1)


class AxisAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = int(embed_dim)  #特征向量的维度，比如 64 个电极通道
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
        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(self.drop(attn), v).transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        return self.drop(self.out_proj(out))

    def forward(self, x: torch.Tensor, attend_dim: int, embed_axis: int) -> torch.Tensor:
        batch, c_dim, k_dim, t_dim = x.shape
        if embed_axis == 1 and attend_dim == 2:
            return self._attention(x.permute(0, 3, 2, 1).reshape(batch * t_dim, k_dim, c_dim)).reshape(batch, t_dim, k_dim, c_dim).permute(0, 3, 2, 1)
        if embed_axis == 1 and attend_dim == 3:
            return self._attention(x.permute(0, 2, 3, 1).reshape(batch * k_dim, t_dim, c_dim)).reshape(batch, k_dim, t_dim, c_dim).permute(0, 3, 1, 2)
        if embed_axis == 2 and attend_dim == 1:
            return self._attention(x.permute(0, 3, 1, 2).reshape(batch * t_dim, c_dim, k_dim)).reshape(batch, t_dim, c_dim, k_dim).permute(0, 2, 3, 1)
        if embed_axis == 2 and attend_dim == 3:
            return self._attention(x.permute(0, 1, 3, 2).reshape(batch * c_dim, t_dim, k_dim)).reshape(batch, c_dim, t_dim, k_dim).permute(0, 1, 3, 2)
        if embed_axis == 3 and attend_dim == 1:
            return self._attention(x.permute(0, 2, 1, 3).reshape(batch * k_dim, c_dim, t_dim)).reshape(batch, k_dim, c_dim, t_dim).permute(0, 2, 1, 3)
        if embed_axis == 3 and attend_dim == 2:
            return self._attention(x.reshape(batch * c_dim, k_dim, t_dim)).reshape(batch, c_dim, k_dim, t_dim)
        raise ValueError("Unsupported attention configuration")


class TriAxisMixerBlock(nn.Module):
    def __init__(self, channel_dim: int, k_dim: int, t_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, channel_dim)
        self.channel_mlp = AxisMLP(channel_dim, dropout)
        self.k_mlp = AxisMLP(k_dim, dropout)
        self.t_mlp = AxisMLP(t_dim, dropout)
        
        self.attn_k_for_c = AxisAttention(channel_dim, n_heads, dropout)
        self.attn_t_for_c = AxisAttention(channel_dim, n_heads, dropout)
        self.attn_c_for_k = AxisAttention(k_dim, n_heads, dropout)
        self.attn_t_for_k = AxisAttention(k_dim, n_heads, dropout)
        self.attn_c_for_t = AxisAttention(t_dim, n_heads, dropout)
        self.attn_k_for_t = AxisAttention(t_dim, n_heads, dropout)
        self.fusion_logits = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        channel_branch = self.channel_mlp(x_norm, dim=1)
        channel_branch = 0.5 * (self.attn_k_for_c(channel_branch, attend_dim=2, embed_axis=1) + self.attn_t_for_c(channel_branch, attend_dim=3, embed_axis=1))
        k_branch = self.k_mlp(x_norm, dim=2)
        k_branch = 0.5 * (self.attn_c_for_k(k_branch, attend_dim=1, embed_axis=2) + self.attn_t_for_k(k_branch, attend_dim=3, embed_axis=2))
        t_branch = self.t_mlp(x_norm, dim=3)
        t_branch = 0.5 * (self.attn_c_for_t(t_branch, attend_dim=1, embed_axis=3) + self.attn_k_for_t(t_branch, attend_dim=2, embed_axis=3))
        weights = torch.softmax(self.fusion_logits, dim=0)
        return x + weights[0] * channel_branch + weights[1] * k_branch + weights[2] * t_branch


class TriAxisEncoder(nn.Module):
    def __init__(self, channel_dim: int, k_dim: int, t_dim: int, n_layers: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([TriAxisMixerBlock(channel_dim, k_dim, t_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BasisMixer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(_get_config(configs, "seq_len", 1))
        self.in_channels = int(_get_config(configs, "enc_in", 64))
        self.num_class = int(_get_config(configs, "num_class", 0))
        self.output_mode = str(_get_config(configs, "output_mode", "classification")).lower()
        self.patch_len = max(1, int(_get_config(configs, "patch_len", 16)))
        stride_cfg = _get_config(configs, "patch_stride", None)
        self.patch_stride = self.patch_len if stride_cfg is None else max(1, int(stride_cfg))
        self.use_channel_adapter = bool(_get_config(configs, "use_channel_adapter", False))
        self.use_channel_prior = bool(_get_config(configs, "use_channel_prior", True))
        self.canonical_channels = int(_get_config(configs, "canonical_channels", 64))
        self.channel_basis_dim = int(_get_config(configs, "channel_basis_dim", min(self.canonical_channels, 64)))
        self.k_basis_dim = int(_get_config(configs, "k_basis_dim", min(self.patch_len, 16)))
        self.t_basis_dim = int(_get_config(configs, "t_basis_dim", 16))
        self.n_layers = max(1, int(_get_config(configs, "t_layer", 3)))
        self.n_heads = max(1, int(_get_config(configs, "n_heads", 4)))
        self.dropout = float(_get_config(configs, "dropout", 0.0))
        self.patch_embed_dim = int(_get_config(configs, "patch_embed_dim", self.channel_basis_dim))

        canonical_coords = _get_config(configs, "canonical_channel_coords", None)
        canonical_names = _get_config(configs, "canonical_channel_names", None)
        canonical_coord_path = _get_config(configs, "canonical_channel_coord_path", None)
        if canonical_coord_path is not None:
            canonical_coords, canonical_names = load_channel_coordinate_csv(str(canonical_coord_path), expected_channels=self.canonical_channels, channel_order=canonical_names)
        elif canonical_coords is not None:
            canonical_coords = torch.as_tensor(canonical_coords, dtype=torch.float32)
        if canonical_coords is None:
            canonical_coords = fibonacci_sphere(self.canonical_channels)

        self.channel_adapter = None
        active_channels = self.in_channels
        if self.use_channel_adapter:
            self.channel_adapter = ChannelAdapter(canonical_channels=self.canonical_channels, use_prior=self.use_channel_prior, sigma=float(_get_config(configs, "channel_adapter_sigma", 0.35)), residual_hidden_dim=int(_get_config(configs, "channel_adapter_hidden", 32)), residual_scale_init=float(_get_config(configs, "channel_adapter_residual_scale", 0.10)), canonical_coords=canonical_coords, canonical_channel_names=canonical_names)
            active_channels = self.canonical_channels
        # if self.channel_basis_dim > active_channels:
        #     raise ValueError("channel_basis_dim cannot exceed active channel count")
        self.channel_basis = LinearChannelProjection(active_channels, self.channel_basis_dim)
        self.k_basis = LinearKProjection(self.patch_len, self.k_basis_dim)
        # self.t_basis = PoolTProjection(self.t_basis_dim)
        patch_num = self._calc_num_patches(self.seq_len)
        # self.t_basis = LinearTProjection(patch_num, self.t_basis_dim)
        self.t_basis = ConvTProjection(
            out_t_dim=self.t_basis_dim,
            kernel_size=int(_get_config(configs, "t_conv_kernel", 5)),
            dropout=self.dropout,
        )

        aug_specs = [s.strip() for s in str(_get_config(configs, "augmentations", "none")).split(",") if s.strip()]
        if not aug_specs:
            aug_specs = ["none"]
        self.augmentations = nn.ModuleList([get_augmentation(spec) for spec in aug_specs])
        self.encoder = TriAxisEncoder(self.channel_basis_dim, self.k_basis_dim, self.t_basis_dim, n_layers=self.n_layers, n_heads=self.n_heads, dropout=self.dropout)
        self.token_proj = None if self.patch_embed_dim == self.channel_basis_dim else nn.Linear(self.channel_basis_dim, self.patch_embed_dim)
        self.head = None if self.num_class <= 0 else nn.Sequential(nn.LayerNorm(self.patch_embed_dim), nn.Linear(self.patch_embed_dim, self.num_class))

    def _calc_num_patches(self, seq_len: int) -> int:
        if seq_len <= self.patch_len:
            return 1
        return int(math.ceil((seq_len - self.patch_len) / self.patch_stride)) + 1

    def _patchify(self, x_3d: torch.Tensor, seq_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        x_patch = x_3d.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
        x_4d = x_patch.permute(0, 1, 3, 2).contiguous()
        if seq_mask is not None:
            patch_mask = seq_mask.unfold(dimension=1, size=self.patch_len, step=self.patch_stride).any(dim=-1)
        else:
            patch_mask = torch.ones(x_4d.size(0), x_4d.size(-1), device=x_4d.device, dtype=torch.bool)
        return x_4d, patch_mask

    def _apply_token_projection(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return token_embeddings if self.token_proj is None else self.token_proj(token_embeddings)

    def _pool_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return token_embeddings.mean(dim=1)

    def forward(self, x_enc: torch.Tensor, channel_coords: Optional[torch.Tensor] = None, channel_mask: Optional[torch.Tensor] = None, seq_mask: Optional[torch.Tensor] = None, return_patch_embeddings: Optional[bool] = None):
        # print(x_enc.shape)
        x = x_enc.transpose(1, 2)
        if self.training and len(self.augmentations) > 0:
            x = self.augmentations[random.randint(0, len(self.augmentations) - 1)](x)
        if self.channel_adapter is not None:
            x = self.channel_adapter(x, channel_coords=channel_coords, channel_mask=channel_mask)
        x, patch_mask = self._patchify(x, seq_mask=seq_mask)
        x = self.channel_basis(x)
        x = self.k_basis(x)
        x = self.t_basis(x)
        x = self.encoder(x)
        token_embeddings = self._apply_token_projection(x.mean(dim=2).transpose(1, 2).contiguous())
        mode = self.output_mode if return_patch_embeddings is None else ("patch" if return_patch_embeddings else "classification")
        if mode == "patch":
            return token_embeddings
        if self.head is None:
            return {"patch_embeddings": token_embeddings, "patch_mask": None, "logits": None} if mode == "both" else token_embeddings
        logits = self.head(self._pool_tokens(token_embeddings))
        return {"patch_embeddings": token_embeddings, "patch_mask": None, "logits": logits} if mode == "both" else logits


Model = BasisMixer


if __name__ == "__main__":
    class DummyConfig:
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

    cfg = DummyConfig()
    model = Model(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    seq_mask = torch.ones(2, cfg.seq_len, dtype=torch.bool)
    out = model(x, seq_mask=seq_mask)
    print(out["patch_embeddings"].shape, out["logits"].shape)
