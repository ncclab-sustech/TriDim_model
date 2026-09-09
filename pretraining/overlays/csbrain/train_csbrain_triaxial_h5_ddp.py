import argparse
import ast
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from CSBrain_TriAxial import CSBrainTriAxial
from pretrain_csbrain_triaxial_h5 import (
    build_csbrain19_montage,
    build_sequential5_montage,
    build_standard1020_21_montage,
    print_model_parameters,
)
from pretrain_trainer_h5 import generate_mask
from pretraining_dataset_h5 import PretrainingDatasetH5


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return distributed, rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_mean(total_loss: float, steps: int, device: torch.device, distributed: bool) -> float:
    stats = torch.tensor([total_loss, float(steps)], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    denom = max(float(stats[1].item()), 1.0)
    return float(stats[0].item() / denom)


def build_model(params):
    if params.montage == "csbrain19":
        if params.n_channels != 19:
            raise ValueError("--montage csbrain19 requires --n_channels 19")
        brain_regions, sorted_indices, topology, electrode_labels = build_csbrain19_montage()
    elif params.montage == "standard1020_21":
        if params.n_channels != 21:
            raise ValueError("--montage standard1020_21 requires --n_channels 21")
        brain_regions, sorted_indices, topology, electrode_labels = build_standard1020_21_montage()
    else:
        brain_regions, sorted_indices, topology, electrode_labels = build_sequential5_montage(params.n_channels)

    print(f"[INFO] electrode_labels: {electrode_labels}", flush=True)
    print(f"[INFO] brain_regions: {brain_regions}", flush=True)
    print(f"[INFO] sorted_indices: {sorted_indices}", flush=True)
    print(f"[INFO] sorted_labels: {[electrode_labels[i] for i in sorted_indices]}", flush=True)
    print(f"[INFO] topology: {topology}", flush=True)

    tem_kernel_sizes = ast.literal_eval(params.TemEmbed_kernel_sizes)
    model = CSBrainTriAxial(
        params.in_dim,
        params.out_dim,
        params.d_model,
        params.dim_feedforward,
        params.seq_len,
        params.n_layer,
        params.nhead,
        tem_kernel_sizes,
        brain_regions,
        sorted_indices,
        dropout=params.dropout,
        layer_scale_init=params.layer_scale_init,
        drop_path_c=params.drop_path_c,
        drop_path_k=params.drop_path_k,
        drop_path_t=params.drop_path_t,
        drop_path_mlp=params.drop_path_mlp,
        drop_path_schedule=params.drop_path_schedule,
    )
    return model, sorted_indices


def parse_args():
    parser = argparse.ArgumentParser(description="DDP CSBrain TriAxial H5 pretraining")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--in_dim", type=int, default=200)
    parser.add_argument("--out_dim", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=200)
    parser.add_argument("--dim_feedforward", type=int, default=800)
    parser.add_argument("--seq_len", type=int, default=30)
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
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--source_n_channels", type=int, default=21)
    parser.add_argument("--channel_indices", type=str, default="all")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--cache_open_files", action="store_true")
    parser.add_argument("--max_open_files", type=int, default=4)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--TemEmbed_kernel_sizes", type=str, default="[(1,), (3,), (5,)]")
    parser.add_argument("--montage", type=str, default="standard1020_21", choices=["csbrain19", "standard1020_21", "sequential5"])
    parser.add_argument("--skip_model_summary", action="store_true")
    parser.add_argument("--save_last", action="store_true")
    return parser.parse_args()


def main():
    params = parse_args()
    distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setup_seed(params.seed + rank)

    if is_main(rank):
        print(params, flush=True)
        print(f"[INFO] DDP world_size={world_size} local_rank={local_rank}", flush=True)

    dataset = PretrainingDatasetH5(
        dataset_dir=params.dataset_dir,
        source_n_channels=params.source_n_channels,
        patch_num=params.seq_len,
        patch_size=params.in_dim,
        channel_indices=params.channel_indices,
        window_stride=params.window_stride,
        recursive=params.recursive,
        max_subjects=params.max_subjects,
        cache_open_files=params.cache_open_files,
        max_open_files=params.max_open_files,
    )
    if dataset.n_channels != params.n_channels:
        raise ValueError(
            f"n_channels mismatch: dataset selected {dataset.n_channels}, "
            f"but --n_channels={params.n_channels}. Check --channel_indices."
        )
    if is_main(rank):
        print(f"[INFO] dataset windows: {len(dataset)}", flush=True)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(params.num_workers > 0),
        drop_last=True,
    )

    model, sorted_indices = build_model(params)
    if is_main(rank) and not params.skip_model_summary:
        print_model_parameters(model)
    model = model.to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    criterion = torch.nn.MSELoss(reduction="mean").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    steps_per_epoch = max(1, len(data_loader))
    if params.lr_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, params.epochs * steps_per_epoch), eta_min=1e-5
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {params.lr_scheduler}")

    model_dir = Path(params.model_dir)
    if is_main(rank):
        model_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    sorted_idx = torch.as_tensor(sorted_indices, dtype=torch.long, device=device)
    best_loss = float("inf")

    for epoch in range(params.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        steps = 0
        iterator = tqdm(data_loader, mininterval=10, disable=not is_main(rank))

        for x in iterator:
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True) / float(params.input_scale)
            target_x = x.index_select(dim=1, index=sorted_idx)

            if params.need_mask:
                bz, ch_num, patch_num, _ = x.shape
                raw_mask = generate_mask(bz, ch_num, patch_num, mask_ratio=params.mask_ratio, device=device)
                model_mask = raw_mask.index_select(dim=1, index=sorted_idx)
                y = model(x, mask=model_mask)
                loss = criterion(y[model_mask], target_x[model_mask])
            else:
                y = model(x)
                loss = criterion(y, target_x)

            if not torch.isfinite(loss):
                if is_main(rank):
                    print("[WARN] non-finite loss detected; skipping step", flush=True)
                continue

            loss.backward()
            if params.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_value)
            optimizer.step()
            scheduler.step()

            loss_value = float(loss.detach().cpu())
            total_loss += loss_value
            steps += 1
            if is_main(rank):
                iterator.set_description(f"epoch={epoch + 1} loss={loss_value:.6f}")

        mean_loss = reduce_mean(total_loss, steps, device, distributed)
        learning_rate = optimizer.param_groups[0]["lr"]
        if is_main(rank):
            print(f"Epoch {epoch + 1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.8f}", flush=True)
            unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
            if mean_loss < best_loss:
                model_path = model_dir / f"epoch{epoch + 1}_loss{mean_loss:.6f}.pth"
                torch.save(unwrapped.state_dict(), model_path)
                print("model save in " + str(model_path), flush=True)
                best_loss = mean_loss
            if params.save_last:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": unwrapped.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "loss": mean_loss,
                    },
                    model_dir / "last.pt",
                )
        if distributed:
            dist.barrier()

    if is_main(rank):
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        torch.save(unwrapped.state_dict(), model_dir / "final_model.pth")
        print("model save in " + str(model_dir / "final_model.pth"), flush=True)

    if hasattr(dataset, "close"):
        dataset.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
