from data_provider.data_loader import ADHDLoader, APAVALoader, ADFTDLoader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch
import numpy as np
import random


def _seed_worker(worker_id):
    """Seed numpy/random inside each DataLoader worker for reproducibility.

    PyTorch already sets torch's per-worker seed via base_seed + worker_id.
    We mirror that into numpy and Python's random so any worker-side RNG
    (augmentations, on-the-fly resampling, etc.) is also deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# All datasets in this release share the same subject-wise h5 layout and are
# routed through ADHDLoader. The loader auto-detects the subject naming scheme
# (sub_*.h5 / sub-*.h5 / S*.h5 / A01T.h5 etc.) so adding a new dataset is just
# a matter of dropping a yaml in configs/datasets/ and registering the key here.
data_dict = {
    "ADHD": ADHDLoader,
    "AD65": ADHDLoader,
    "ADFTD": ADFTDLoader,
    "APAVA": APAVALoader,
    "BCIC2A": ADHDLoader,
    "Broderick": ADHDLoader,
    "ChineseEEG1": ADHDLoader,
    "EEGMAT": ADHDLoader,
    "Exoskeleton_WalkStop": ADHDLoader,
    "FACED_new": ADHDLoader,
    "ISRUC-Sleep_1": ADHDLoader,
    "ISRUC_S1": ADHDLoader,
    "MDD": ADHDLoader,
    "Physionet_MI": ADHDLoader,
    "RestCog": ADHDLoader,
    "SEED": ADHDLoader,
    "SEEDIV": ADHDLoader,
    "SEED_V": ADHDLoader,
    "SEED_VIG": ADHDLoader,
    "SHU": ADHDLoader,
    "SleepEDF_full": ADHDLoader,
    "sleep-cassette-200hz": ADHDLoader,
}


def data_provider(args, flag):
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
