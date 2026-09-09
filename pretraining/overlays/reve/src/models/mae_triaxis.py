from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from models.encoder import FourierEmb4D, mlp_pos_embedding, patch_embedding
from utils.initialization import ConfigInit, init_mae


class TriAxisMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        from tridim import TriAxisEncoder

        self.config = config
        self.masking_ratio = float(config.decoder.masking.ratio)
        assert 0.0 < self.masking_ratio < 1.0, "masking ratio must be kept between 0 and 1"

        self.embed_dim = int(config.encoder.transformer.embed_dim)
        self.patch_size = int(config.encoder.patch_size)
        self.overlap_size = int(config.encoder.patch_overlap)
        self.noise_ratio = float(config.encoder.noise_ratio)
        self.freqs = int(config.encoder.freqs)
        self.n_channels = int(config.data.n_channels)
        self.window_size = int(config.data.window_size)
        self.n_time_patches = (self.window_size - self.patch_size) // (self.patch_size - self.overlap_size) + 1

        self.to_patch_embedding = patch_embedding(self.embed_dim, self.patch_size)
        self.fourier4d = FourierEmb4D(self.embed_dim, freqs=self.freqs)
        self.mlp4d = mlp_pos_embedding(self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)

        self.encoder = TriAxisEncoder(
            channel_dim=self.n_channels,
            k_dim=self.n_time_patches,
            t_dim=self.embed_dim,
            n_layers=int(config.encoder.transformer.depth),
            n_heads=int(config.encoder.transformer.heads),
            dropout=float(config.triaxis.dropout),
            layer_scale_init=float(config.triaxis.layer_scale_init),
            drop_path_c=float(config.triaxis.drop_path_c),
            drop_path_k=float(config.triaxis.drop_path_k),
            drop_path_t=float(config.triaxis.drop_path_t),
            drop_path_mlp=float(config.triaxis.drop_path_mlp),
            drop_path_schedule=str(config.triaxis.drop_path_schedule),
        )

        decoder_dim = int(config.decoder.transformer.embed_dim)
        self.decoder = TriAxisEncoder(
            channel_dim=self.n_channels,
            k_dim=self.n_time_patches,
            t_dim=decoder_dim,
            n_layers=int(config.decoder.transformer.depth),
            n_heads=int(config.decoder.transformer.heads),
            dropout=float(config.triaxis.dropout),
            layer_scale_init=float(config.triaxis.layer_scale_init),
            drop_path_c=float(config.triaxis.drop_path_c),
            drop_path_k=float(config.triaxis.drop_path_k),
            drop_path_t=float(config.triaxis.drop_path_t),
            drop_path_mlp=float(config.triaxis.drop_path_mlp),
            drop_path_schedule=str(config.triaxis.drop_path_schedule),
        )
        self.decoder.dim = decoder_dim

        self.encoder_mask_token = nn.Parameter(torch.randn(self.embed_dim))
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.enc_to_dec = nn.Linear(self.embed_dim, decoder_dim) if self.embed_dim != decoder_dim else nn.Identity()
        self.pos_enc_to_dec = nn.Linear(self.embed_dim, decoder_dim) if self.embed_dim != decoder_dim else nn.Identity()
        self.to_pixels = nn.Linear(decoder_dim, self.patch_size)

        self.token_avg = bool(config.token_avg)
        if self.token_avg:
            self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.cls_to_pixels = nn.Sequential(
                nn.Linear(self.embed_dim, 4 * self.embed_dim, bias=False),
                nn.ReLU(),
                nn.Linear(4 * self.embed_dim, self.patch_size),
            )
            self.token_avg_lambda = float(config.token_avg_lambda)

        self.init_weights()

    def init_weights(self):
        config_megatron = ConfigInit(**self.config.init)
        init_mae(self, config_megatron)
        nn.init.trunc_normal_(self.encoder_mask_token, std=0.02)
        print("TriAxis MAE weights initialized")

    def _make_random_indices(self, batch: int, num_patches: int, device: torch.device):
        num_masked = int(self.masking_ratio * num_patches)
        if self.training:
            rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(42)
            rand_indices = torch.rand(batch, num_patches, device=device, generator=generator).argsort(dim=-1)
        return rand_indices[:, :num_masked], rand_indices[:, num_masked:]

    def forward(self, eeg, pos, b_m=None, b_u=None, return_patches=False):  # noqa: PLR0915
        device = eeg.device
        patches = eeg.unfold(
            dimension=2,
            size=self.patch_size,
            step=self.patch_size - self.overlap_size,
        )
        b, c, h, p = patches.shape
        if c != self.n_channels or h != self.n_time_patches:
            raise RuntimeError(
                f"TriAxisMAE expected grid [{self.n_channels}, {self.n_time_patches}], got [{c}, {h}]"
            )

        patches_flat = rearrange(patches, "b c h e -> b (c h) e", c=c, h=h, e=p)
        num_patches = c * h
        if self.training:
            noise = np.random.normal(loc=0, scale=self.noise_ratio, size=(c, 3))
            pos = pos + torch.from_numpy(noise).to(pos)

        pos4d = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos4d) + self.mlp4d(pos4d))
        token_flat = self.to_patch_embedding(patches_flat) + pos_embed

        if b_m is None:
            masked_indices, unmasked_indices = self._make_random_indices(b, num_patches, device)
        else:
            masked_indices, unmasked_indices = b_m, b_u
        num_masked = masked_indices.shape[1]
        batch_range = torch.arange(b, device=device)[:, None]
        masked_patches = patches_flat[batch_range, masked_indices]

        masked_encoder_tokens = token_flat.clone()
        encoder_mask = repeat(self.encoder_mask_token, "d -> b n d", b=b, n=num_masked)
        masked_encoder_tokens[batch_range, masked_indices] = encoder_mask + pos_embed[batch_range, masked_indices]
        encoder_grid = rearrange(masked_encoder_tokens, "b (c h) d -> b c h d", c=c, h=h)

        if self.token_avg and self.training:
            layer_outputs = self.encoder(encoder_grid, return_all_layers=True)
            encoded_grid = layer_outputs[-1]
            key_value_tokens = torch.cat(
                [rearrange(layer, "b c h d -> b (c h) d") for layer in layer_outputs],
                dim=1,
            )
            query_output = self.cls_query_token.expand(b, -1, -1)
            attention_scores = torch.matmul(query_output, key_value_tokens.transpose(-1, -2)) / (self.embed_dim**0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context = torch.matmul(attention_weights, key_value_tokens).squeeze(1)
        else:
            encoded_grid = self.encoder(encoder_grid)
            context = None

        encoded_flat = rearrange(encoded_grid, "b c h d -> b (c h) d")
        decoder_tokens = self.enc_to_dec(encoded_flat)
        decoder_pos_emb = self.pos_enc_to_dec(pos_embed).to(pos_embed)
        decoder_tokens = decoder_tokens + decoder_pos_emb
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=b, n=num_masked)
        decoder_tokens[batch_range, masked_indices] = mask_tokens + decoder_pos_emb[batch_range, masked_indices]

        decoder_grid = rearrange(decoder_tokens, "b (c h) d -> b c h d", c=c, h=h)
        decoded_grid = self.decoder(decoder_grid)
        decoded_flat = rearrange(decoded_grid, "b c h d -> b (c h) d")
        decoded_mask_tokens = decoded_flat[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(decoded_mask_tokens)
        loss = F.l1_loss(pred_pixel_values, masked_patches)

        if self.token_avg and self.training and context is not None:
            repeated_context = repeat(context, "b d -> b n d", n=num_masked) + pos_embed[batch_range, masked_indices]
            cls_pixels = self.cls_to_pixels(repeated_context)
            loss_cls = F.l1_loss(cls_pixels, masked_patches)
            loss = loss + self.token_avg_lambda * loss_cls

        if return_patches:
            return loss, pred_pixel_values, masked_patches
        return loss
