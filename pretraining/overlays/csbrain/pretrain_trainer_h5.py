"""Portable masked-reconstruction trainer for the CSBrain H5 integration.

This reconstructs the original training objective using public upstream
utilities. Result provenance remains checkpoint-based; this file is not
presented as an original run log.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss
from tqdm import tqdm

from utils.util import generate_mask


class Trainer:
    def __init__(self, params, data_loader, model):
        self.params = params
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{params.cuda}")
        else:
            self.device = torch.device("cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction="mean").to(self.device)

        if params.parallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay,
        )
        steps = max(1, params.epochs * len(data_loader))
        if params.lr_scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=steps, eta_min=1e-5
            )
        elif params.lr_scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif params.lr_scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, 5 * len(data_loader)), gamma=0.5
            )
        elif params.lr_scheduler == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    10 * len(data_loader),
                    20 * len(data_loader),
                    30 * len(data_loader),
                ],
                gamma=0.1,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {params.lr_scheduler}")

    @property
    def base_model(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def train(self):
        model_dir = Path(self.params.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")

        for epoch in range(self.params.epochs):
            self.model.train()
            losses = []
            for x in tqdm(self.data_loader, mininterval=10):
                x = x.to(self.device, non_blocking=True) / self.params.input_scale
                target = x[:, self.base_model.sorted_indices, :, :]

                self.optimizer.zero_grad(set_to_none=True)
                if self.params.need_mask:
                    batch, channels, patches, _ = target.shape
                    mask = generate_mask(
                        batch,
                        channels,
                        patches,
                        mask_ratio=self.params.mask_ratio,
                        device=self.device,
                    )
                    prediction = self.model(x, mask=mask)
                    loss = self.criterion(prediction[mask == 1], target[mask == 1])
                else:
                    prediction = self.model(x)
                    loss = self.criterion(prediction, target)

                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.params.clip_value
                    )
                self.optimizer.step()
                self.scheduler.step()
                losses.append(float(loss.detach().cpu()))

            mean_loss = float(np.mean(losses))
            learning_rate = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}: Training Loss: {mean_loss:.6f}, "
                f"Learning Rate: {learning_rate:.6f}",
                flush=True,
            )
            if mean_loss < best_loss:
                checkpoint = model_dir / f"epoch{epoch + 1}_loss{mean_loss:.6f}.pth"
                torch.save(self.model.state_dict(), checkpoint)
                print(f"Model saved at {checkpoint}", flush=True)
                best_loss = mean_loss
