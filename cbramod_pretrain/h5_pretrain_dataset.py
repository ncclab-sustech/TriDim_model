from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleRef:
    file_path: str
    trial_name: str
    segment_name: str
    start: int
    signal_len: int


def subject_id_from_path(file_path: str) -> str:
    return Path(file_path).stem


def infer_ct_layout(shape: Sequence[int], n_channels: int) -> Optional[str]:
    if len(shape) != 2:
        return None
    if shape[0] == n_channels:
        return "CT"
    if shape[1] == n_channels:
        return "TC"
    return None


def scan_h5_files(
    data_root: str,
    n_channels: int,
    window_size: int,
    window_stride: int,
    recursive: bool = True,
    max_subjects: Optional[int] = None,
) -> List[SampleRef]:
    root = Path(data_root).expanduser().resolve()

    if recursive:
        file_paths = sorted(
            str(p)
            for p in root.rglob("*")
            if p.is_file() and p.name.startswith("sub_") and p.suffix.lower() == ".h5"
        )
    else:
        file_paths = sorted(
            str(p)
            for p in root.iterdir()
            if p.is_file() and p.name.startswith("sub_") and p.suffix.lower() == ".h5"
        )

    if max_subjects is not None and int(max_subjects) > 0:
        file_paths = file_paths[: int(max_subjects)]

    if not file_paths:
        raise FileNotFoundError(f"No sub_*.h5 files found under: {root}")

    print(f"[INFO] scanned subject files: {len(file_paths)}", flush=True)

    samples: List[SampleRef] = []
    for fp in file_paths:
        try:
            with h5py.File(fp, "r") as f:
                for trial_name in f.keys():
                    trial_group = f[trial_name]
                    for segment_name in trial_group.keys():
                        seg_group = trial_group[segment_name]
                        if "eeg" not in seg_group:
                            continue
                        eeg_ds = seg_group["eeg"]
                        layout = infer_ct_layout(eeg_ds.shape, n_channels)
                        if layout is None:
                            continue
                        signal_len = int(eeg_ds.shape[1] if layout == "CT" else eeg_ds.shape[0])
                        if signal_len < window_size:
                            continue
                        for start in range(0, signal_len - window_size + 1, window_stride):
                            samples.append(
                                SampleRef(
                                    file_path=fp,
                                    trial_name=str(trial_name),
                                    segment_name=str(segment_name),
                                    start=int(start),
                                    signal_len=signal_len,
                                )
                            )
        except Exception as e:
            print(f"[WARN] skipping unreadable H5: {fp} | {type(e).__name__}: {e}", flush=True)

    if not samples:
        raise RuntimeError("No valid windows were generated from the scanned H5 files.")

    print(f"[INFO] total windows: {len(samples)}", flush=True)
    return samples


def split_samples(
    samples: Sequence[SampleRef],
    val_ratio: float,
    split_by: str,
    seed: int,
) -> Tuple[List[SampleRef], List[SampleRef]]:
    import random

    split_by = str(split_by).lower()
    if val_ratio <= 0.0:
        return list(samples), []

    if split_by == "sample":
        samples = list(samples)
        rng = random.Random(seed)
        rng.shuffle(samples)
        n_val = max(1, int(round(len(samples) * val_ratio))) if len(samples) > 1 else 0
        return samples[n_val:], samples[:n_val]

    if split_by == "file":
        units = sorted({s.file_path for s in samples})
        key_fn = lambda s: s.file_path
    elif split_by == "subject":
        units = sorted({subject_id_from_path(s.file_path) for s in samples})
        key_fn = lambda s: subject_id_from_path(s.file_path)
    else:
        raise ValueError(f"Unsupported split_by: {split_by}")

    rng = random.Random(seed)
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * val_ratio))) if len(units) > 1 else 0
    val_set = set(units[:n_val])

    train_samples = [s for s in samples if key_fn(s) not in val_set]
    val_samples = [s for s in samples if key_fn(s) in val_set]

    if not train_samples:
        train_samples = list(samples)
        val_samples = list(samples[: min(len(samples), 1)])
    return train_samples, val_samples


class H5PretrainDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleRef],
        n_channels: int,
        seq_len: int,
        normalize_per_window: bool = False,
        scale_divisor: float = 100.0,
        cache_open_files: bool = True,
        max_open_files: int = 4,
    ):
        self.samples = list(samples)
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.normalize_per_window = bool(normalize_per_window)
        self.scale_divisor = float(scale_divisor)
        self.cache_open_files = bool(cache_open_files)
        self.max_open_files = max(1, int(max_open_files))
        self._h5_cache: "OrderedDict[str, h5py.File]" = OrderedDict()
        if not self.samples:
            raise RuntimeError("Dataset received an empty sample list.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        for f in self._h5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_h5_file(self, file_path: str) -> h5py.File:
        if not self.cache_open_files:
            return h5py.File(file_path, "r")

        f = self._h5_cache.pop(file_path, None)
        if f is not None:
            self._h5_cache[file_path] = f
            return f

        while len(self._h5_cache) >= self.max_open_files:
            _, old_f = self._h5_cache.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass

        f = h5py.File(file_path, "r")
        self._h5_cache[file_path] = f
        return f

    def _read_window_ct(self, ref: SampleRef) -> np.ndarray:
        if self.cache_open_files:
            f = self._get_h5_file(ref.file_path)
            eeg_ds = f[ref.trial_name][ref.segment_name]["eeg"]
            layout = infer_ct_layout(eeg_ds.shape, self.n_channels)
            if layout is None:
                raise ValueError(
                    f"Unexpected eeg shape {eeg_ds.shape} in {ref.file_path}:{ref.trial_name}/{ref.segment_name}"
                )
            if layout == "CT":
                x = eeg_ds[:, ref.start : ref.start + self.seq_len]
                return np.asarray(x, dtype=np.float32)
            x = eeg_ds[ref.start : ref.start + self.seq_len, :]
            return np.asarray(x, dtype=np.float32).T

        with h5py.File(ref.file_path, "r") as f:
            eeg_ds = f[ref.trial_name][ref.segment_name]["eeg"]
            layout = infer_ct_layout(eeg_ds.shape, self.n_channels)
            if layout is None:
                raise ValueError(
                    f"Unexpected eeg shape {eeg_ds.shape} in {ref.file_path}:{ref.trial_name}/{ref.segment_name}"
                )
            if layout == "CT":
                x = eeg_ds[:, ref.start : ref.start + self.seq_len]
                return np.asarray(x, dtype=np.float32)
            x = eeg_ds[ref.start : ref.start + self.seq_len, :]
            return np.asarray(x, dtype=np.float32).T

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ref = self.samples[idx]
        x = self._read_window_ct(ref)  # [C, T]
        if x.shape[0] != self.n_channels:
            raise RuntimeError(f"Channel mismatch: got {x.shape[0]}, expected {self.n_channels}")
        if x.shape[1] != self.seq_len:
            raise RuntimeError(f"Seq length mismatch: got {x.shape[1]}, expected {self.seq_len}")

        x = torch.from_numpy(x.copy()).float()  # [C, T]
        if self.normalize_per_window:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std
        else:
            x = x / self.scale_divisor

        x = x.transpose(0, 1).contiguous()  # [T, C]
        seq_mask = torch.ones(self.seq_len, dtype=torch.bool)
        return {
            "eeg": x,
            "seq_mask": seq_mask,
        }
