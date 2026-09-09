from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", "180"))
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
    return distributed, rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def load_model_state(model: torch.nn.Module, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    has_module = any(k.startswith("module.") for k in state_dict)
    if has_module:
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    else:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def atomic_torch_save(obj, path: Path, retries: int = 2, retry_delay: float = 15.0) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(retries + 1):
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{attempt}")
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return True
        except Exception as exc:
            last_error = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            if attempt < retries:
                print(
                    f"[WARN] failed to save {path} on attempt {attempt + 1}/{retries + 1}: {exc}; retrying",
                    flush=True,
                )
                time.sleep(retry_delay)
    print(f"[WARN] failed to save {path}: {last_error}", flush=True)
    return False


def build_config(args: argparse.Namespace):
    return ns(
        token_avg=args.token_avg,
        token_avg_lambda=args.token_avg_lambda,
        init={
            "init_method": "full_megatron",
            "init_std": 0.02,
            "init_cutoff_factor": 3.0,
            "hidden_size": args.embed_dim,
            "num_hidden_layers": args.encoder_depth + args.decoder_depth,
            "init_cls": True,
        },
        encoder=ns(
            transformer=ns(
                embed_dim=args.embed_dim,
                depth=args.encoder_depth,
                heads=args.heads,
                head_dim=args.head_dim,
                mlp_dim_ratio=args.mlp_dim_ratio,
                use_geglu=args.use_geglu,
            ),
            freqs=args.freqs,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap,
            noise_ratio=args.noise_ratio,
        ),
        decoder=ns(
            transformer=ns(
                embed_dim=args.decoder_embed_dim,
                depth=args.decoder_depth,
                heads=args.heads,
                head_dim=args.head_dim,
                mlp_dim_ratio=args.mlp_dim_ratio,
                use_geglu=args.use_geglu,
            ),
            masking=ns(ratio=args.mask_ratio),
        ),
        data=ns(n_channels=args.n_channels, window_size=args.window_size),
        triaxis=ns(
            dropout=args.triaxis_dropout,
            layer_scale_init=args.layer_scale_init,
            drop_path_c=args.drop_path_c,
            drop_path_k=args.drop_path_k,
            drop_path_t=args.drop_path_t,
            drop_path_mlp=args.drop_path_mlp,
            drop_path_schedule=args.drop_path_schedule,
        ),
    )


def summarize_parameters(model: torch.nn.Module, name: str = "model", rank: int = 0) -> None:
    if not is_main_process(rank):
        return
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[INFO] {name} params: total={total:,} trainable={trainable:,} fp32_size={total * 4 / (1024 ** 2):.2f} MB",
        flush=True,
    )


def build_dataset(args: argparse.Namespace):
    from utils.h5_tuh_dataset import (
        H5TuhPretrainDataset,
        STANDARD1020_21,
        load_channel_positions,
        scan_h5_files,
        subject_id_from_path,
    )

    samples = scan_h5_files(
        data_root=args.dataset_dir,
        n_channels=args.n_channels,
        window_size=args.window_size,
        window_stride=args.window_stride,
        recursive=args.recursive,
        max_subjects=args.max_subjects,
    )
    positions = load_channel_positions(args.coord_csv, STANDARD1020_21)
    ds = H5TuhPretrainDataset(
        samples=samples,
        positions=positions,
        n_channels=args.n_channels,
        seq_len=args.window_size,
        normalize_per_window=args.normalize_per_window,
        scale_divisor=args.input_scale,
        clip=args.clip,
        cache_open_files=args.cache_open_files,
        max_open_files=args.max_open_files,
    )
    stats = {
        "n_total_windows": len(samples),
        "n_subjects": len({subject_id_from_path(s.file_path) for s in samples}),
    }
    return ds, stats


def build_model(args: argparse.Namespace, config):
    if args.triaxis_module_dir and args.triaxis_module_dir not in sys.path:
        sys.path.append(args.triaxis_module_dir)

    if args.variant == "reve":
        from models.mae import MAE

        return MAE(config)
    if args.variant == "triaxis":
        from models.mae_triaxis import TriAxisMAE

        return TriAxisMAE(config)
    raise ValueError(f"Unknown variant: {args.variant}")


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_loss: float, args: argparse.Namespace) -> bool:
    return atomic_torch_save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_loss": float(best_loss),
            "args": vars(args),
        },
        path,
        retries=args.checkpoint_retries,
        retry_delay=args.checkpoint_retry_delay,
    )


def train(args: argparse.Namespace) -> None:
    distributed, rank, local_rank, world_size = setup_distributed()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cuda_index = local_rank if distributed else args.cuda
    device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(cuda_index)

    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    ds, stats = build_dataset(args)
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
        if distributed
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    if is_main_process(rank):
        print(args, flush=True)
        print(f"[INFO] distributed: {distributed} rank={rank} world_size={world_size}", flush=True)
        print(f"[INFO] dataset windows: {stats['n_total_windows']}", flush=True)
        print(f"[INFO] subjects: {stats['n_subjects']}", flush=True)
        print(f"[INFO] batches_per_epoch: {len(loader)}", flush=True)
        print(f"[INFO] per_gpu_batch_size: {args.batch_size}", flush=True)
        print(f"[INFO] global_batch_size: {args.batch_size * world_size}", flush=True)
        print(f"[INFO] device: {device}", flush=True)

    config = build_config(args)
    model = build_model(args, config).to(device)

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_model_state(model, checkpoint["model"])

    if distributed:
        model = DDP(model, device_ids=[cuda_index] if device.type == "cuda" else None)
    summarize_parameters(model, f"reve_{args.variant}", rank=rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * len(loader)),
        eta_min=args.min_lr,
    )

    start_epoch = 0
    best_loss = float("inf")
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))
        if is_main_process(rank):
            print(f"[INFO] resumed from {args.resume} at epoch {start_epoch + 1}", flush=True)

    set_seed(args.seed + rank)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    init_time = time.time()
    try:
        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            losses = []
            pbar = tqdm(loader, mininterval=30, disable=(args.disable_tqdm or not is_main_process(rank)))
            for step, (eeg, pos) in enumerate(pbar, start=1):
                eeg = eeg.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=args.amp and device.type == "cuda",
                ):
                    loss = model(eeg, pos)
                finite = torch.isfinite(loss).all()
                if distributed:
                    finite_flag = torch.tensor(1 if bool(finite) else 0, device=device)
                    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                    finite = bool(finite_flag.item())
                if not finite:
                    if is_main_process(rank):
                        print("[WARN] non-finite loss detected; skipping step", flush=True)
                    continue
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                loss_value = float(reduce_mean(loss, world_size).cpu())
                losses.append(loss_value)
                status = (
                    f"variant={args.variant} epoch={epoch + 1} step={step}/{len(loader)} "
                    f"loss={loss_value:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                if args.disable_tqdm:
                    if step == 1 or step % args.log_every == 0 or step == len(loader):
                        if is_main_process(rank):
                            print(status, flush=True)
                else:
                    if is_main_process(rank):
                        pbar.set_description(status)

            mean_loss = float(np.mean(losses)) if losses else math.inf
            if is_main_process(rank):
                print(
                    f"Epoch {epoch + 1}: Training Loss: {mean_loss:.6f}, Learning Rate: {optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )
                if mean_loss < best_loss:
                    best_path = output_dir / f"epoch{epoch + 1}_loss{mean_loss:.6f}.pth"
                    if atomic_torch_save(
                        unwrap_model(model).state_dict(),
                        best_path,
                        retries=args.checkpoint_retries,
                        retry_delay=args.checkpoint_retry_delay,
                    ):
                        best_loss = mean_loss
                        print(f"model save in {best_path}", flush=True)
                if not save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_loss, args):
                    print("[WARN] last checkpoint save failed; keeping previous last.pt", flush=True)
            if distributed:
                dist.barrier()

        if is_main_process(rank):
            raw_model = unwrap_model(model)
            atomic_torch_save(
                raw_model.state_dict(),
                output_dir / "final_model.pth",
                retries=args.checkpoint_retries,
                retry_delay=args.checkpoint_retry_delay,
            )
            if hasattr(raw_model, "encoder"):
                try:
                    atomic_torch_save(
                        raw_model.encoder.state_dict(),
                        output_dir / "encoder.pth",
                        retries=args.checkpoint_retries,
                        retry_delay=args.checkpoint_retry_delay,
                    )
                except Exception:
                    pass
            print(f"[INFO] finished in {time.time() - init_time:.1f}s", flush=True)
    finally:
        if hasattr(ds, "close"):
            ds.close()
        cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REVE H5 pretraining on TUH-style merged H5 files")
    parser.add_argument("--variant", choices=["reve", "triaxis"], default="reve")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--coord_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max_subjects", type=int, default=2000)
    parser.add_argument("--window_size", type=int, default=6000)
    parser.add_argument("--window_stride", type=int, default=6000)
    parser.add_argument("--n_channels", type=int, default=21)
    parser.add_argument("--input_scale", type=float, default=100.0)
    parser.add_argument("--clip", type=float, default=15.0)
    parser.add_argument("--normalize_per_window", action="store_true")
    parser.add_argument("--cache_open_files", action="store_true")
    parser.add_argument("--max_open_files", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_depth", type=int, default=22)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--mlp_dim_ratio", type=float, default=2.66)
    parser.add_argument("--use_geglu", action="store_true", default=True)
    parser.add_argument("--no_geglu", dest="use_geglu", action="store_false")
    parser.add_argument("--freqs", type=int, default=4)
    parser.add_argument("--noise_ratio", type=float, default=0.0025)
    parser.add_argument("--patch_size", type=int, default=200)
    parser.add_argument("--patch_overlap", type=int, default=20)
    parser.add_argument("--mask_ratio", type=float, default=0.55)
    parser.add_argument("--token_avg", action="store_true", default=True)
    parser.add_argument("--no_token_avg", dest="token_avg", action="store_false")
    parser.add_argument("--token_avg_lambda", type=float, default=0.1)
    parser.add_argument("--triaxis_module_dir", type=str, default=None)
    parser.add_argument("--triaxis_dropout", type=float, default=0.1)
    parser.add_argument("--layer_scale_init", type=float, default=1e-2)
    parser.add_argument("--drop_path_c", type=float, default=0.0)
    parser.add_argument("--drop_path_k", type=float, default=0.0)
    parser.add_argument("--drop_path_t", type=float, default=0.0)
    parser.add_argument("--drop_path_mlp", type=float, default=0.0)
    parser.add_argument("--drop_path_schedule", choices=["linear", "constant"], default="linear")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint_retries", type=int, default=2)
    parser.add_argument("--checkpoint_retry_delay", type=float, default=15.0)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
