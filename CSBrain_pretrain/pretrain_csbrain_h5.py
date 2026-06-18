import argparse
import ast
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from pretraining_dataset_h5 import PretrainingDatasetH5
from models.CSBrain import CSBrain
from pretrain_trainer_h5 import Trainer


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("=" * 80)
    print(f"[MODEL] Total parameters:        {total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"[MODEL] Trainable parameters:    {trainable_params:,} ({trainable_params / 1e6:.3f} M)")
    print(f"[MODEL] Non-trainable parameters:{non_trainable_params:,} ({non_trainable_params / 1e6:.3f} M)")
    print("=" * 80)


def sort_indices_by_topology(
    brain_regions: List[int], electrode_labels: List[str], topology: Dict[int, List[str]]
) -> List[int]:
    """Sort input channel indices first by region id, then by the region-specific topology order."""
    region_groups = {}
    for i, region in enumerate(brain_regions):
        region_groups.setdefault(region, []).append((i, electrode_labels[i]))

    sorted_indices = []
    for region in sorted(region_groups.keys()):
        region_electrodes = region_groups[region]
        sorted_electrodes = sorted(region_electrodes, key=lambda x: topology[region].index(x[1]))
        sorted_indices.extend([e[0] for e in sorted_electrodes])
    return sorted_indices


def build_csbrain19_montage() -> Tuple[List[int], List[int], Dict[int, List[str]], List[str]]:
    """Official CSBrain/TUH 19-channel 10-20 style montage.

    Region ids follow the original CSBrain convention:
      0 frontal, 1 parietal, 2 temporal, 3 occipital, 4 central.
    """
    brain_regions = [
        0, 0, 0, 0, 4, 4, 1, 1, 3, 3,
        0, 0, 2, 2, 2, 2, 0, 4, 1,
    ]
    electrode_labels = [
        "FP1-REF", "FP2-REF", "F3-REF", "F4-REF", "C3-REF", "C4-REF",
        "P3-REF", "P4-REF", "O1-REF", "O2-REF", "F7-REF", "F8-REF",
        "T3-REF", "T4-REF", "T5-REF", "T6-REF", "FZ-REF", "CZ-REF", "PZ-REF",
    ]
    topology = {
        0: ["FP1-REF", "F7-REF", "F3-REF", "FZ-REF", "F4-REF", "F8-REF", "FP2-REF"],
        1: ["P3-REF", "PZ-REF", "P4-REF"],
        2: ["T3-REF", "T5-REF", "T6-REF", "T4-REF"],
        3: ["O1-REF", "O2-REF"],
        4: ["C3-REF", "CZ-REF", "C4-REF"],
    }
    sorted_indices = sort_indices_by_topology(brain_regions, electrode_labels, topology)
    return brain_regions, sorted_indices, topology, electrode_labels


def build_standard1020_21_montage() -> Tuple[List[int], List[int], Dict[int, List[str]], List[str]]:
    """21-channel standard 10-20 montage used by our H5 pretraining files.

    Expected source channel order, derived from input21_from_standard1020.csv:
      FP1, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8,
      P7, P3, PZ, P4, P8, O1, O2, A1, A2.

    We keep CSBrain's five anatomical region ids:
      0 frontal, 1 parietal, 2 temporal/lateral, 3 occipital, 4 central.

    A1/A2 are auricular/reference-like electrodes, not cortical 10-20 scalp sites.
    For full 21-channel pretraining we place them in the temporal/lateral group so
    the model can consume all channels without changing CSBrain internals.
    """
    electrode_labels = [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ", "C4", "T8",
        "P7", "P3", "PZ", "P4", "P8", "O1", "O2", "A1", "A2",
    ]

    brain_regions = [
        0, 0, 0, 0, 0, 0, 0,  # FP1, FP2, F7, F3, FZ, F4, F8
        2,                    # T7
        4, 4, 4,              # C3, CZ, C4
        2,                    # T8
        2,                    # P7, treated as posterior temporal/lateral like T5
        1, 1, 1,              # P3, PZ, P4
        2,                    # P8, treated as posterior temporal/lateral like T6
        3, 3,                 # O1, O2
        2, 2,                 # A1, A2, auricular/lateral
    ]

    topology = {
        # left-anterior -> midline -> right-anterior, matching CSBrain's frontal convention
        0: ["FP1", "F7", "F3", "FZ", "F4", "F8", "FP2"],
        # left -> midline -> right
        1: ["P3", "PZ", "P4"],
        # left inferior/lateral -> posterior lateral -> right lateral/inferior.
        # This mimics CSBrain's T3,T5,T6,T4 order while retaining A1/A2.
        2: ["A1", "T7", "P7", "P8", "T8", "A2"],
        3: ["O1", "O2"],
        4: ["C3", "CZ", "C4"],
    }
    sorted_indices = sort_indices_by_topology(brain_regions, electrode_labels, topology)
    return brain_regions, sorted_indices, topology, electrode_labels


def build_sequential5_montage(n_channels: int) -> Tuple[List[int], List[int], Dict[int, List[str]], List[str]]:
    groups = np.array_split(np.arange(n_channels), 5)
    brain_regions = [0] * n_channels
    for region_id, group in enumerate(groups):
        for ch in group.tolist():
            brain_regions[ch] = int(region_id)
    electrode_labels = [f"CH{i}" for i in range(n_channels)]
    sorted_indices = list(range(n_channels))
    topology = {i: [f"CH{j}" for j in groups[i].tolist()] for i in range(len(groups))}
    return brain_regions, sorted_indices, topology, electrode_labels


def main():
    parser = argparse.ArgumentParser(description="CSBrain pretraining on local H5 EEG windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--parallel", action="store_true")

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
    parser.add_argument("--dim_feedforward", type=int, default=800)
    parser.add_argument("--seq_len", type=int, default=30, help="number of patches")
    parser.add_argument("--n_channels", type=int, default=21, help="selected/model channel count")

    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--need_mask", action="store_true", default=True)
    parser.add_argument("--no_mask", dest="need_mask", action="store_false")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--input_scale", type=float, default=100.0)

    parser.add_argument("--dataset_dir", type=str, required=True, help="root dir of H5 files")
    parser.add_argument("--source_n_channels", type=int, default=21, help="channel count stored in the source H5")
    parser.add_argument(
        "--channel_indices",
        type=str,
        default="all",
        help="selected source channel indices; use 'all' to keep all channels",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--cache_open_files", action="store_true")
    parser.add_argument("--max_open_files", type=int, default=4)

    parser.add_argument("--model_dir", type=str, default="model_dir")
    parser.add_argument("--TemEmbed_kernel_sizes", type=str, default="[(1,), (3,), (5,)]")
    parser.add_argument(
        "--montage",
        type=str,
        default="standard1020_21",
        choices=["csbrain19", "standard1020_21", "sequential5"],
    )
    parser.add_argument("--skip_model_summary", action="store_true")
    parser.add_argument("--skip_flops", action="store_true", default=True)
    parser.add_argument("--run_sanity_check", action="store_true")

    params = parser.parse_args()
    print(params, flush=True)
    setup_seed(params.seed)

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

    print(f"[INFO] dataset windows: {len(dataset)}", flush=True)

    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(params.num_workers > 0),
        drop_last=True,
    )

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
    model = CSBrain(
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
    )

    print_model_parameters(model)

    if params.run_sanity_check:
        device = torch.device(f"cuda:{params.cuda}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        xb = next(iter(data_loader)).to(device) / params.input_scale
        mask = torch.zeros(xb.shape[0], xb.shape[1], xb.shape[2], dtype=torch.bool, device=device)
        with torch.no_grad():
            yb = model(xb, mask=mask.index_select(1, torch.as_tensor(sorted_indices, device=device)))
        print(f"[SANITY] x={tuple(xb.shape)} y={tuple(yb.shape)}", flush=True)
        return

    trainer = Trainer(params, data_loader, model)
    trainer.train()

    if hasattr(dataset, "close"):
        dataset.close()


if __name__ == "__main__":
    main()
