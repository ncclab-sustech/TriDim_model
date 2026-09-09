from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


STANDARD1020_21 = [
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
    "A1",
    "A2",
]


@dataclass(frozen=True)
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


def open_h5_read(file_path: str) -> h5py.File:
    try:
        return h5py.File(file_path, "r", locking=False)
    except TypeError:
        return h5py.File(file_path, "r")


def scan_h5_files(
    data_root: str,
    n_channels: int,
    window_size: int,
    window_stride: int,
    recursive: bool = True,
    max_subjects: Optional[int] = None,
) -> list[SampleRef]:
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
    samples: list[SampleRef] = []
    total_files = len(file_paths)
    for file_idx, fp in enumerate(file_paths, start=1):
        if file_idx == 1 or file_idx % 100 == 0 or file_idx == total_files:
            print(
                f"[INFO] indexing H5 file {file_idx}/{total_files}: {Path(fp).name} windows={len(samples)}",
                flush=True,
            )
        try:
            with open_h5_read(fp) as f:
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
        except Exception as exc:
            print(f"[WARN] skipping unreadable H5: {fp} | {type(exc).__name__}: {exc}", flush=True)

    if not samples:
        raise RuntimeError("No valid windows were generated from the scanned H5 files.")
    print(f"[INFO] total windows: {len(samples)}", flush=True)
    return samples


def load_channel_positions(csv_path: str, channel_order: Sequence[str]) -> torch.Tensor:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        lookup = {
            str(row["channel"]).strip().upper(): [float(row["x"]), float(row["y"]), float(row["z"])]
            for row in reader
        }
    missing = [name for name in channel_order if name.upper() not in lookup]
    if missing:
        raise ValueError(f"Missing channel coordinates in {csv_path}: {missing}")
    return torch.tensor([lookup[name.upper()] for name in channel_order], dtype=torch.float32)


class H5TuhPretrainDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleRef],
        positions: torch.Tensor,
        n_channels: int,
        seq_len: int,
        normalize_per_window: bool = False,
        scale_divisor: float = 100.0,
        clip: float = 15.0,
        cache_open_files: bool = True,
        max_open_files: int = 4,
    ):
        self.samples = list(samples)
        self.positions = positions.float()
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.normalize_per_window = bool(normalize_per_window)
        self.scale_divisor = float(scale_divisor)
        self.clip = float(clip)
        self.cache_open_files = bool(cache_open_files)
        self.max_open_files = max(1, int(max_open_files))
        self._h5_cache: OrderedDict[str, h5py.File] = OrderedDict()
        if not self.samples:
            raise RuntimeError("Dataset received an empty sample list.")
        if self.positions.shape != (self.n_channels, 3):
            raise ValueError(f"Expected positions [{self.n_channels}, 3], got {tuple(self.positions.shape)}")

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

        f = open_h5_read(file_path)
        self._h5_cache[file_path] = f
        return f

    def _read_window_ct(self, ref: SampleRef) -> np.ndarray:
        if self.cache_open_files:
            f = self._get_h5_file(ref.file_path)
            return self._read_from_file(f, ref)

        with open_h5_read(ref.file_path) as f:
            return self._read_from_file(f, ref)

    def _read_from_file(self, f: h5py.File, ref: SampleRef) -> np.ndarray:
        eeg_ds = f[ref.trial_name][ref.segment_name]["eeg"]
        layout = infer_ct_layout(eeg_ds.shape, self.n_channels)
        if layout is None:
            raise ValueError(
                f"Unexpected eeg shape {eeg_ds.shape} in "
                f"{ref.file_path}:{ref.trial_name}/{ref.segment_name}"
            )
        if layout == "CT":
            x = eeg_ds[:, ref.start : ref.start + self.seq_len]
            return np.asarray(x, dtype=np.float32)
        x = eeg_ds[ref.start : ref.start + self.seq_len, :]
        return np.asarray(x, dtype=np.float32).T

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x = self._read_window_ct(self.samples[idx])
        if x.shape != (self.n_channels, self.seq_len):
            raise RuntimeError(f"Expected EEG [{self.n_channels}, {self.seq_len}], got {tuple(x.shape)}")

        eeg = torch.from_numpy(x.copy()).float()
        if self.normalize_per_window:
            mean = eeg.mean(dim=1, keepdim=True)
            std = eeg.std(dim=1, keepdim=True).clamp_min(1e-6)
            eeg = (eeg - mean) / std
        else:
            eeg = eeg / self.scale_divisor
        if self.clip > 0:
            eeg = eeg.clamp(-self.clip, self.clip)
        return eeg.contiguous(), self.positions.clone()
