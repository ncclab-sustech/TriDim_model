import os
from typing import Optional

import numpy as np
import torch
from torch.nn import MSELoss
from tqdm import tqdm


def generate_mask(bz: int, ch_num: int, patch_num: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """Generate a boolean random patch mask with shape [B, C, patch_num]."""
    num_mask = int(round(float(mask_ratio) * patch_num))
    num_mask = max(1, min(patch_num, num_mask))
    noise = torch.rand(bz, ch_num, patch_num, device=device)
    ids = torch.argsort(noise, dim=-1)[:, :, :num_mask]
    mask = torch.zeros(bz, ch_num, patch_num, dtype=torch.bool, device=device)
    mask.scatter_(dim=-1, index=ids, value=True)
    return mask




class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction="mean").to(self.device)

        if self.params.parallel:
            device_ids = list(range(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)
        os.makedirs(self.params.model_dir, exist_ok=True)

        if not getattr(self.params, "skip_model_summary", False):
            try:
                from torchinfo import summary

                summary(
                    self.model,
                    input_size=(1, self.params.n_channels, self.params.seq_len, self.params.in_dim),
                )
            except Exception as e:
                print(f"[WARN] torchinfo summary failed: {type(e).__name__}: {e}", flush=True)

        if not getattr(self.params, "skip_flops", True):
            try:
                from ptflops import get_model_complexity_info

                macs, n_params = get_model_complexity_info(
                    self.model,
                    (self.params.n_channels, self.params.seq_len, self.params.in_dim),
                    as_strings=True,
                    print_per_layer_stat=False,
                    verbose=False,
                )
                print(f"Computational complexity: {macs}", flush=True)
                print(f"Number of parameters: {n_params}", flush=True)
            except Exception as e:
                print(f"[WARN] ptflops failed: {type(e).__name__}: {e}", flush=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay
        )

        scheduler_name = self.params.lr_scheduler
        if scheduler_name == "CosineAnnealingLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.params.epochs * self.data_length), eta_min=1e-5
            )
        elif scheduler_name == "ExponentialLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999999999)
        elif scheduler_name == "StepLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, 5 * self.data_length), gamma=0.5
            )
        elif scheduler_name == "MultiStepLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[10 * self.data_length, 20 * self.data_length, 30 * self.data_length],
                gamma=0.1,
            )
        elif scheduler_name == "CyclicLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=1e-6,
                max_lr=0.001,
                step_size_up=max(1, self.data_length * 5),
                step_size_down=max(1, self.data_length * 2),
                mode="exp_range",
                gamma=0.9,
                cycle_momentum=False,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def _sorted_index_tensor(self) -> Optional[torch.Tensor]:
        sorted_indices = getattr(self._unwrap_model(), "sorted_indices", None)
        if sorted_indices is None or len(sorted_indices) == 0:
            return None
        return torch.as_tensor(sorted_indices, dtype=torch.long, device=self.device)

    def train(self):
        best_loss = float("inf")
        sorted_idx = self._sorted_index_tensor()

        for epoch in range(self.params.epochs):
            self.model.train()
            losses = []
            pbar = tqdm(self.data_loader, mininterval=10)

            for x in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                x = x.to(self.device, non_blocking=True) / float(self.params.input_scale)

                if sorted_idx is not None:
                    target_x = x.index_select(dim=1, index=sorted_idx)
                else:
                    target_x = x

                if self.params.need_mask:
                    bz, ch_num, patch_num, _ = x.shape
                    raw_mask = generate_mask(
                        bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device
                    )
                    model_mask = raw_mask.index_select(dim=1, index=sorted_idx) if sorted_idx is not None else raw_mask
                    y = self.model(x, mask=model_mask)
                    loss = self.criterion(y[model_mask], target_x[model_mask])
                else:
                    y = self.model(x)
                    loss = self.criterion(y, target_x)

                if not torch.isfinite(loss):
                    print("[WARN] non-finite loss detected; skipping step", flush=True)
                    continue

                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                pbar.set_description(f"epoch={epoch + 1} loss={loss_value:.6f}")

            mean_loss = float(np.mean(losses)) if losses else float("inf")
            learning_rate = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.8f}", flush=True)

            if mean_loss < best_loss:
                model_path = os.path.join(self.params.model_dir, f"epoch{epoch + 1}_loss{mean_loss:.6f}.pth")
                state_dict = self._unwrap_model().state_dict()
                torch.save(state_dict, model_path)
                print("model save in " + model_path, flush=True)
                best_loss = mean_loss
