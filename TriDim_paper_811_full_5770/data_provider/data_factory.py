from data_provider.data_loader import EEGDatasetLoader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch
import numpy as np
import random


def _seed_worker(worker_id):
    """Seed numpy/random inside each DataLoader worker for reproducibility.

    PyTorch already sets torch's per-worker seed via base_seed + worker_id.
    We mirror that into numpy and Python's random so any worker-side RNG
    (e.g., on-the-fly resampling) is also deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# The eight paper datasets share the HDF5/Zarr loader, which detects the
# supported subject and session naming schemes and applies the configured
# subject-aware split.
data_dict = {
    "AD65": EEGDatasetLoader,
    "BCIC2A": EEGDatasetLoader,
    "FACED_new": EEGDatasetLoader,
    "Physionet_MI": EEGDatasetLoader,
    "SEED": EEGDatasetLoader,
    "SEED_V": EEGDatasetLoader,
    "SHU": EEGDatasetLoader,
    "SleepEDF_full": EEGDatasetLoader,
}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise KeyError(
            f"Unsupported dataset {args.data!r}; expected one of {sorted(data_dict)}"
        )
    Data = data_dict[args.data]

    flag_upper = str(flag).upper()
    # Keep eval deterministic: only training split should shuffle.
    shuffle_flag = flag_upper == "TRAIN"
    batch_size = args.batch_size
    drop_last = False
    data_set = Data(
        root_path=args.root_path,
        args=args,
        flag=flag,
    )

    # Per-loader generator seeded from the current torch RNG (which run.py
    # seeds per-itr via torch.manual_seed). This makes the shuffle order
    # bit-exact reproducible across runs.
    g = torch.Generator()
    g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()) & 0x7FFFFFFFFFFFFFFF)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    return data_set, data_loader
