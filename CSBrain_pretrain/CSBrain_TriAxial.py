
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CSBrain import *  # TemEmbedEEGLayer, BrainEmbedEEGLayer
from collections import defaultdict


# =============================================================================
# Tri-axial core (minimal V11 block only)
# =============================================================================

class InstanceTimeNorm(nn.Module):
    """
    Per-sample, per-channel normalization along the time axis.

    For CSBrain inputs [B, C, P, K], we flatten patches to [B, C, P*K],
    normalize over the last axis, then reshape back. This mirrors the V11
    idea of doing input normalization before patchification / patch embedding.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"InstanceTimeNorm expects [B,C,P,K], got {tuple(x.shape)}")
        b, c, p, k = x.shape
        x_flat = x.reshape(b, c, p * k)
        mean = x_flat.mean(dim=-1, keepdim=True)
        var = x_flat.var(dim=-1, keepdim=True, unbiased=False)
        x_flat = (x_flat - mean) / torch.sqrt(var + self.eps)
        return x_flat.reshape(b, c, p, k)



class DropPath(nn.Module):
    """Per-sample stochastic depth."""
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


class AxisRMSNorm(nn.Module):
    """RMSNorm applied along a specific axis of a 4D tensor [B, C, K, T]."""
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
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (
            torch.matmul(self.drop(attn), v)
            .transpose(1, 2)
            .reshape(batch, seq_len, self.embed_dim)
        )
        return self.drop(self.out_proj(out))

    def forward(self, x: torch.Tensor, attend_dim: int, embed_axis: int) -> torch.Tensor:
        batch, c_dim, k_dim, t_dim = x.shape
        if embed_axis == 1 and attend_dim == 2:
            return (
                self._attention(x.permute(0, 3, 2, 1).reshape(batch * t_dim, k_dim, c_dim))
                .reshape(batch, t_dim, k_dim, c_dim)
                .permute(0, 3, 2, 1)
            )
        if embed_axis == 1 and attend_dim == 3:
            return (
                self._attention(x.permute(0, 2, 3, 1).reshape(batch * k_dim, t_dim, c_dim))
                .reshape(batch, k_dim, t_dim, c_dim)
                .permute(0, 3, 1, 2)
            )
        if embed_axis == 2 and attend_dim == 1:
            return (
                self._attention(x.permute(0, 3, 1, 2).reshape(batch * t_dim, c_dim, k_dim))
                .reshape(batch, t_dim, c_dim, k_dim)
                .permute(0, 2, 3, 1)
            )
        if embed_axis == 2 and attend_dim == 3:
            return (
                self._attention(x.permute(0, 1, 3, 2).reshape(batch * c_dim, t_dim, k_dim))
                .reshape(batch, c_dim, t_dim, k_dim)
                .permute(0, 1, 3, 2)
            )
        if embed_axis == 3 and attend_dim == 1:
            return (
                self._attention(x.permute(0, 2, 1, 3).reshape(batch * k_dim, c_dim, t_dim))
                .reshape(batch, k_dim, c_dim, t_dim)
                .permute(0, 2, 1, 3)
            )
        if embed_axis == 3 and attend_dim == 2:
            return self._attention(x.reshape(batch * c_dim, k_dim, t_dim)).reshape(batch, c_dim, k_dim, t_dim)
        raise ValueError("Unsupported attention configuration")


class TriAxisMixerBlock(nn.Module):
    """
    Minimal V11 tri-axial block.

    We keep only the tri-axial mixer core:
      - per-axis RMSNorm
      - shared cross-axis attention
      - per-axis MLP
      - dual residual with LayerScale
      - optional DSS DropPath (defaults can be 0)

    Input / output shape: [B, D, C, P]
      D = embedding dim
      C = EEG channels
      P = patch number
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

        self.norm_c_attn = AxisRMSNorm(self.channel_dim, axis=1)
        self.norm_k_attn = AxisRMSNorm(self.k_dim, axis=2)
        self.norm_t_attn = AxisRMSNorm(self.t_dim, axis=3)
        self.norm_c_mlp = AxisRMSNorm(self.channel_dim, axis=1)
        self.norm_k_mlp = AxisRMSNorm(self.k_dim, axis=2)
        self.norm_t_mlp = AxisRMSNorm(self.t_dim, axis=3)

        self.attn_for_c = AxisAttention(self.channel_dim, n_heads=n_heads, dropout=dropout)
        self.attn_for_k = AxisAttention(self.k_dim, n_heads=n_heads, dropout=dropout)
        self.attn_for_t = AxisAttention(self.t_dim, n_heads=n_heads, dropout=dropout)

        self.channel_mlp = AxisMLP(self.channel_dim, dropout=dropout)
        self.k_mlp = AxisMLP(self.k_dim, dropout=dropout)
        self.t_mlp = AxisMLP(self.t_dim, dropout=dropout)

        self.attn_fusion_logits = nn.Parameter(torch.zeros(3))
        self.mlp_fusion_logits = nn.Parameter(torch.zeros(3))

        ls = float(layer_scale_init)
        self.gamma_attn = nn.Parameter(ls * torch.ones(1, self.channel_dim, 1, 1))
        self.gamma_mlp = nn.Parameter(ls * torch.ones(1, self.channel_dim, 1, 1))

        self.dp_c_attn = DropPath(drop_path_c)
        self.dp_k_attn = DropPath(drop_path_k)
        self.dp_t_attn = DropPath(drop_path_t)
        self.dp_mlp = DropPath(drop_path_mlp)

    def _attn_branch(self, x: torch.Tensor) -> torch.Tensor:
        x_c = self.norm_c_attn(x)
        c_out = 0.5 * (
            self.attn_for_c(x_c, attend_dim=2, embed_axis=1)
            + self.attn_for_c(x_c, attend_dim=3, embed_axis=1)
        )
        c_out = self.dp_c_attn(c_out)

        x_k = self.norm_k_attn(x)
        k_out = 0.5 * (
            self.attn_for_k(x_k, attend_dim=1, embed_axis=2)
            + self.attn_for_k(x_k, attend_dim=3, embed_axis=2)
        )
        k_out = self.dp_k_attn(k_out)

        x_t = self.norm_t_attn(x)
        t_out = 0.5 * (
            self.attn_for_t(x_t, attend_dim=1, embed_axis=3)
            + self.attn_for_t(x_t, attend_dim=2, embed_axis=3)
        )
        t_out = self.dp_t_attn(t_out)

        w = torch.softmax(self.attn_fusion_logits, dim=0)
        return w[0] * c_out + w[1] * k_out + w[2] * t_out

    def _mlp_branch(self, x: torch.Tensor) -> torch.Tensor:
        x_c = self.norm_c_mlp(x)
        c_out = self.channel_mlp(x_c, dim=1)

        x_k = self.norm_k_mlp(x)
        k_out = self.k_mlp(x_k, dim=2)

        x_t = self.norm_t_mlp(x)
        t_out = self.t_mlp(x_t, dim=3)

        w = torch.softmax(self.mlp_fusion_logits, dim=0)
        fused = w[0] * c_out + w[1] * k_out + w[2] * t_out
        return self.dp_mlp(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma_attn * self._attn_branch(x)
        x = x + self.gamma_mlp * self._mlp_branch(x)
        return x


class TriAxisEncoder(nn.Module):
    """Stack of TriAxisMixerBlocks with optional DSS schedule."""
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
                    drop_path_c * s,
                    drop_path_k * s,
                    drop_path_t * s,
                    drop_path_mlp * s,
                ))
        else:
            rates = [(drop_path_c, drop_path_k, drop_path_t, drop_path_mlp)] * n_layers

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
        self.num_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Original CSBrain pieces kept intact
# =============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(d_model // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, mask_x.shape[1] // 2 + 1)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def generate_area_config(brain_regions):
    region_to_channels = defaultdict(list)
    for channel_idx, region in enumerate(brain_regions):
        region_to_channels[region].append(channel_idx)

    area_config = {}
    for region, channels in region_to_channels.items():
        area_config[f'region_{region}'] = {
            'channels': len(channels),
            'slice': slice(channels[0], channels[-1] + 1)
        }
    return area_config


class CSBrainTriAxial(nn.Module):
    """
    CSBrain skeleton + tri-axial mixer replacement.

    What is kept from CSBrain:
      - channel sorting by montage topology
      - PatchEmbedding
      - TemEmbedEEGLayer
      - BrainEmbedEEGLayer
      - patch-wise reconstruction target
      - proj_out = Linear(d_model -> out_dim)

    What is replaced:
      - original CSBrain_TransformerEncoderLayer stack
      - replaced by tri-axial mixer blocks operating on [B, D, C, P]
        after a simple permutation from [B, C, P, D].

    NOTE:
      We intentionally do NOT bring over channel_adapter, fixed interpolation,
      multi-level readout, or the full BasisMixer frontend. This file only
      injects the tri-axial block into the existing CSBrain pretraining setup.
      We do, however, add the V11-style input normalization right before PatchEmbedding.
    """
    def __init__(
        self,
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,   # kept only for CLI compatibility; unused
        seq_len=30,
        n_layer=12,
        nhead=8,
        TemEmbed_kernel_sizes=[(1,), (3,), (5,)],
        brain_regions=None,
        sorted_indices=None,
        dropout=0.1,
        layer_scale_init=1e-2,
        drop_path_c=0.0,
        drop_path_k=0.0,
        drop_path_t=0.0,
        drop_path_mlp=0.0,
        drop_path_schedule="linear",
        use_input_norm=True,
    ):
        super().__init__()
        if brain_regions is None:
            brain_regions = []
        if sorted_indices is None:
            sorted_indices = []

        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        self.TemEmbed_kernel_sizes = TemEmbed_kernel_sizes
        kernel_sizes = self.TemEmbed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim, kernel_sizes=kernel_sizes, stride=1)

        self.brain_regions = brain_regions
        self.area_config = generate_area_config(sorted(brain_regions))
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)
        self.sorted_indices = sorted_indices
        self.seq_len = int(seq_len)
        self.n_layer = int(n_layer)
        self.out_dim = int(out_dim)
        self.d_model = int(d_model)
        self.use_input_norm = bool(use_input_norm)
        self.input_norm = InstanceTimeNorm() if self.use_input_norm else nn.Identity()

        # Tri-axial encoder runs on [B, D, C, P]
        self.encoder = TriAxisEncoder(
            channel_dim=d_model,
            k_dim=len(brain_regions),
            t_dim=seq_len,
            n_layers=n_layer,
            n_heads=nhead,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
            drop_path_c=drop_path_c,
            drop_path_k=drop_path_k,
            drop_path_t=drop_path_t,
            drop_path_mlp=drop_path_mlp,
            drop_path_schedule=drop_path_schedule,
        )

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

        self.features_by_layer = []
        self.input_features = []

    def forward(self, x, mask=None):
        # Keep original CSBrain montage sorting
        x = x[:, self.sorted_indices, :, :]

        # Add V11-style input normalization before patch embedding
        x = self.input_norm(x)

        # Original CSBrain patch embedding
        patch_emb = self.patch_embedding(x, mask)  # [B, C, P, D]

        self.features_by_layer = []
        for layer_idx in range(self.encoder.num_layers):
            # Keep original CSBrain temporal EEG embedding + region embedding
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb

            # Only replace the core block with tri-axial mixer
            tri_in = patch_emb.permute(0, 3, 1, 2).contiguous()   # [B, D, C, P]
            tri_out = self.encoder.layers[layer_idx](tri_in)      # [B, D, C, P]
            patch_emb = tri_out.permute(0, 2, 3, 1).contiguous()  # [B, C, P, D]
            self.features_by_layer.append(patch_emb)

        out = self.proj_out(patch_emb)  # [B, C, P, K]
        return out


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CSBrainTriAxial(
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
        seq_len=30, n_layer=12, nhead=8,
        brain_regions=[0] * 21, sorted_indices=list(range(21)),
        drop_path_c=0.0, drop_path_k=0.0, drop_path_t=0.0, drop_path_mlp=0.0,
        use_input_norm=True,
    ).to(device)
    a = torch.randn((8, 21, 30, 200), device=device)
    b = model(a)
    print(a.shape, b.shape)
