
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from basis_pretrain_wrapper_v10 import CBraModV11
from h5_pretrain_dataset import (
    H5PretrainDataset,
    scan_h5_files,
    subject_id_from_path,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize_parameters(model: torch.nn.Module, name: str = "model") -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    param_size_mb = total_params * 4 / (1024 ** 2)
    print(
        f"[INFO] {name} params: total={total_params:,} | "
        f"trainable={trainable_params:,} | frozen={frozen_params:,} | "
        f"fp32_size={param_size_mb:.2f} MB",
        flush=True,
    )


def generate_mask(bz: int, ch_num: int, patch_num: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """CBraMod-style channel-patch mask: [B, C, patch_num]."""
    num_mask = int(round(float(mask_ratio) * patch_num))
    num_mask = max(1, min(patch_num, num_mask))
    noise = torch.rand(bz, ch_num, patch_num, device=device)
    ids = torch.argsort(noise, dim=-1)[:, :, :num_mask]
    mask = torch.zeros(bz, ch_num, patch_num, dtype=torch.bool, device=device)
    mask.scatter_(dim=-1, index=ids, value=True)
    return mask


def is_dist_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_enabled() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_enabled() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def cleanup_distributed() -> None:
    if is_dist_enabled():
        dist.barrier()
        dist.destroy_process_group()


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.cuda)))

    use_ddp = world_size > 1
    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is not available")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.cuda)
            device = torch.device(f"cuda:{args.cuda}")
        else:
            device = torch.device("cpu")
            local_rank = 0
            rank = 0

    return device, rank, world_size, use_ddp


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def build_dataloader(args: argparse.Namespace, device: torch.device, rank: int, world_size: int):
    raw_window_size = int(args.seq_len) * int(args.in_dim)
    samples = scan_h5_files(
        data_root=args.dataset_dir,
        n_channels=args.n_channels,
        window_size=raw_window_size,
        window_stride=args.window_stride if args.window_stride is not None else raw_window_size,
        recursive=args.recursive,
        max_subjects=args.max_subjects,
    )

    ds = H5PretrainDataset(
        samples,
        n_channels=args.n_channels,
        seq_len=raw_window_size,
        normalize_per_window=False,
        scale_divisor=args.input_scale,
        cache_open_files=args.cache_open_files,
        max_open_files=args.max_open_files,
    )

    sampler: Optional[DistributedSampler]
    if world_size > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    stats = {
        "n_total_windows": len(samples),
        "n_subjects": len({subject_id_from_path(s.file_path) for s in samples}),
        "raw_window_size": raw_window_size,
    }
    return ds, loader, sampler, stats


def train(model, loader, sampler, optimizer, scheduler, device, args):
    criterion = MSELoss(reduction="mean").to(device)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        losses = []

        pbar = tqdm(
            loader,
            mininterval=10,
            disable=(not is_main_process()),
        )

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            x_raw = batch["eeg"].to(device, non_blocking=True)   # [B, T, C]
            x_raw = x_raw.transpose(1, 2).contiguous()          # [B, C, T]

            bsz, ch_num, total_t = x_raw.shape
            expected_t = args.seq_len * args.in_dim
            if total_t != expected_t:
                raise RuntimeError(
                    f"Raw window length {total_t} != seq_len*in_dim {expected_t}. "
                    f"Got seq_len={args.seq_len}, in_dim={args.in_dim}."
                )

            x = x_raw.view(bsz, ch_num, args.seq_len, args.in_dim).contiguous()  # [B,C,P,K]

            if args.need_mask:
                mask = generate_mask(
                    bz=bsz,
                    ch_num=ch_num,
                    patch_num=args.seq_len,
                    mask_ratio=args.mask_ratio,
                    device=device,
                )
                y = model(x, mask=mask)
                loss = criterion(y[mask], x[mask])
            else:
                y = model(x)
                loss = criterion(y, x)

            if not torch.isfinite(loss):
                if is_main_process():
                    print("[WARN] non-finite loss detected; skipping step", flush=True)
                continue

            loss.backward()
            if args.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            optimizer.step()
            scheduler.step()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if is_main_process():
                pbar.set_description(f"epoch={epoch + 1} loss={loss_value:.6f}")

        mean_loss_local = float(np.mean(losses)) if losses else float("inf")
        mean_loss_tensor = torch.tensor(mean_loss_local, device=device, dtype=torch.float64)
        if is_dist_enabled():
            dist.all_reduce(mean_loss_tensor, op=dist.ReduceOp.SUM)
            mean_loss = float((mean_loss_tensor / get_world_size()).item())
        else:
            mean_loss = float(mean_loss_tensor.item())

        learning_rate = optimizer.param_groups[0]["lr"]

        if is_main_process():
            print(
                f"Epoch {epoch + 1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.8f}",
                flush=True,
            )

            if mean_loss < best_loss:
                model_path = Path(args.model_dir) / f"epoch{epoch + 1}_loss{mean_loss:.6f}.pth"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(unwrap_model(model).state_dict(), model_path)
                print("model save in " + str(model_path), flush=True)
                best_loss = mean_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CBraMod pretraining with original ACPE frontend + V11 tri-axis block stack (DDP-ready)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--in_dim", type=int, default=200, help="patch size")
    parser.add_argument("--out_dim", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=200)
    parser.add_argument("--dim_feedforward", type=int, default=800, help="kept for CLI compatibility only")
    parser.add_argument("--seq_len", type=int, default=30, help="number of patches")
    parser.add_argument("--n_channels", type=int, default=21)

    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--need_mask", action="store_true", default=True)
    parser.add_argument("--no_mask", dest="need_mask", action="store_false")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--input_scale", type=float, default=100.0)

    parser.add_argument("--layer_scale_init", type=float, default=1e-2)
    parser.add_argument("--drop_path_c", type=float, default=0.0)
    parser.add_argument("--drop_path_k", type=float, default=0.0)
    parser.add_argument("--drop_path_t", type=float, default=0.0)
    parser.add_argument("--drop_path_mlp", type=float, default=0.0)
    parser.add_argument("--drop_path_schedule", type=str, default="linear", choices=["linear", "constant"])

    parser.add_argument("--dataset_dir", type=str, required=True, help="root dir of H5 files")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--cache_open_files", action="store_true")
    parser.add_argument("--max_open_files", type=int, default=4)

    parser.add_argument("--model_dir", type=str, default="model_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device, rank, world_size, use_ddp = setup_distributed(args)

    if is_main_process():
        print(args, flush=True)
        print(f"[INFO] rank={rank} | world_size={world_size} | use_ddp={use_ddp}", flush=True)

    ds, loader, sampler, stats = build_dataloader(args, device, rank, world_size)

    if is_main_process():
        print(f"[INFO] dataset windows: {stats['n_total_windows']}", flush=True)
        print(f"[INFO] subjects: {stats['n_subjects']}", flush=True)
        print(f"[INFO] raw_window_size: {stats['raw_window_size']}", flush=True)

    model = CBraModV11(
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        nhead=args.nhead,
        n_channels=args.n_channels,
        dropout=args.dropout,
        layer_scale_init=args.layer_scale_init,
        drop_path_c=args.drop_path_c,
        drop_path_k=args.drop_path_k,
        drop_path_t=args.drop_path_t,
        drop_path_mlp=args.drop_path_mlp,
        drop_path_schedule=args.drop_path_schedule,
    ).to(device)

    if is_main_process():
        summarize_parameters(model, name="full_model")

    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.lr_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs * len(loader)), eta_min=1e-5
        )
    elif args.lr_scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999999999)
    elif args.lr_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, 5 * len(loader)), gamma=0.5)
    elif args.lr_scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[10 * len(loader), 20 * len(loader), 30 * len(loader)],
            gamma=0.1,
        )
    elif args.lr_scheduler == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-6,
            max_lr=0.001,
            step_size_up=max(1, len(loader) * 5),
            step_size_down=max(1, len(loader) * 2),
            mode='exp_range',
            gamma=0.9,
            cycle_momentum=False,
        )
    else:
        cleanup_distributed()
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")

    try:
        train(model, loader, sampler, optimizer, scheduler, device, args)
    finally:
        if hasattr(ds, "close"):
            ds.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
