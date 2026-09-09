
import torch
import torch.nn as nn
import torch.nn.functional as F

from tridim import TriAxisEncoder


class CBraModV11(nn.Module):
    """
    CBraMod protocol preserved:
      - input: [B, C, P, K]
      - original PatchEmbedding (conv stem + spectral branch + ACPE)
      - original masked reconstruction head shape: [B, C, P, out_dim]

    Only replace the original TransformerEncoder stack with our tri-axis block stack.
    """
    def __init__(
        self,
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,  # kept only for CLI compatibility
        seq_len=30,
        n_layer=12,
        nhead=8,
        n_channels=21,
        dropout=0.1,
        layer_scale_init=1e-2,
        drop_path_c=0.0,
        drop_path_k=0.0,
        drop_path_t=0.0,
        drop_path_mlp=0.0,
        drop_path_schedule="linear",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.n_channels = int(n_channels)

        self.patch_embedding = PatchEmbedding(
            in_dim=in_dim,
            out_dim=out_dim,
            d_model=d_model,
            seq_len=seq_len,
        )

        self.encoder = TriAxisEncoder(
            channel_dim=self.n_channels,
            k_dim=self.seq_len,
            t_dim=self.d_model,
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

        self.sorted_indices = list(range(self.n_channels))
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        patch_emb = self.patch_embedding(x, mask)  # [B, C, P, d_model]
        feats = self.encoder(patch_emb)            # [B, C, P, d_model]
        out = self.proj_out(feats)                 # [B, C, P, out_dim]
        return out


class PatchEmbedding(nn.Module):
    """
    Original CBraMod patch embedding / ACPE front-end preserved.

    Input:
        x    [B, C, P, K]
        mask [B, C, P] or None
    Output:
        patch_emb [B, C, P, d_model]
    """
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = int(d_model)

        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            ),
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
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape

        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask] = self.mask_encoding

        stem_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(stem_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        fft_x = stem_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(fft_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)

        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)
        patch_emb = patch_emb + positional_embedding
        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
