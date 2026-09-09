from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

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


def parse_channel_indices(channel_indices: Optional[str], source_n_channels: int) -> List[int]:
    if channel_indices is None or str(channel_indices).strip() == "" or str(channel_indices).lower() == "all":
        return list(range(int(source_n_channels)))
    indices = [int(x.strip()) for x in str(channel_indices).split(",") if x.strip() != ""]
    if len(indices) == 0:
        raise ValueError("channel_indices is empty. Use 'all' or a comma-separated list such as 0,1,2.")
    bad = [i for i in indices if i < 0 or i >= int(source_n_channels)]
    if bad:
        raise ValueError(f"channel_indices out of range for source_n_channels={source_n_channels}: {bad}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"channel_indices contains duplicates: {indices}")
    return indices


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
    source_n_channels: int,
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
                        layout = infer_ct_layout(eeg_ds.shape, source_n_channels)
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


class PretrainingDatasetH5(Dataset):
    """Sliding-window EEG H5 dataset for CSBrain/CBraMod-style pretraining.

    It reads H5 files with layout [C, T] or [T, C], optionally selects a subset of
    source channels, and returns [model_channels, patch_num, patch_size].
    """

    def __init__(
        self,
        dataset_dir: str,
        source_n_channels: int,
        patch_num: int,
        patch_size: int = 200,
        channel_indices: Optional[str] = None,
        window_stride: Optional[int] = None,
        recursive: bool = True,
        max_subjects: Optional[int] = None,
        cache_open_files: bool = True,
        max_open_files: int = 4,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.source_n_channels = int(source_n_channels)
        self.channel_indices = parse_channel_indices(channel_indices, self.source_n_channels)
        self.n_channels = len(self.channel_indices)
        self.patch_num = int(patch_num)
        self.patch_size = int(patch_size)
        self.window_size = self.patch_num * self.patch_size
        self.window_stride = self.window_size if window_stride is None else int(window_stride)
        self.recursive = bool(recursive)
        self.max_subjects = max_subjects
        self.cache_open_files = bool(cache_open_files)
        self.max_open_files = max(1, int(max_open_files))
        self._h5_cache: "OrderedDict[str, h5py.File]" = OrderedDict()

        self.samples = scan_h5_files(
            data_root=self.dataset_dir,
            source_n_channels=self.source_n_channels,
            window_size=self.window_size,
            window_stride=self.window_stride,
            recursive=self.recursive,
            max_subjects=self.max_subjects,
        )
        print(
            f"[INFO] source_n_channels={self.source_n_channels}, "
            f"selected_n_channels={self.n_channels}, channel_indices={self.channel_indices}",
            flush=True,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_cache"] = OrderedDict()
        return state

    def close(self):
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
            layout = infer_ct_layout(eeg_ds.shape, self.source_n_channels)
            if layout is None:
                raise ValueError(
                    f"Unexpected eeg shape {eeg_ds.shape} in "
                    f"{ref.file_path}:{ref.trial_name}/{ref.segment_name}"
                )
            if layout == "CT":
                x = eeg_ds[:, ref.start : ref.start + self.window_size]
                x = np.asarray(x, dtype=np.float32)
            else:
                x = eeg_ds[ref.start : ref.start + self.window_size, :]
                x = np.asarray(x, dtype=np.float32).T
            return x[self.channel_indices, :]

        with h5py.File(ref.file_path, "r") as f:
            eeg_ds = f[ref.trial_name][ref.segment_name]["eeg"]
            layout = infer_ct_layout(eeg_ds.shape, self.source_n_channels)
            if layout is None:
                raise ValueError(
                    f"Unexpected eeg shape {eeg_ds.shape} in "
                    f"{ref.file_path}:{ref.trial_name}/{ref.segment_name}"
                )
            if layout == "CT":
                x = eeg_ds[:, ref.start : ref.start + self.window_size]
                x = np.asarray(x, dtype=np.float32)
            else:
                x = eeg_ds[ref.start : ref.start + self.window_size, :]
                x = np.asarray(x, dtype=np.float32).T
            return x[self.channel_indices, :]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ref = self.samples[idx]
        x = self._read_window_ct(ref)

        if x.shape[0] != self.n_channels:
            raise RuntimeError(f"Channel count mismatch: got {x.shape[0]}, expected {self.n_channels}")
        if x.shape[1] != self.window_size:
            raise RuntimeError(f"Window length mismatch: got {x.shape[1]}, expected {self.window_size}")

        x = torch.from_numpy(x.copy()).float()
        x = x.view(self.n_channels, self.patch_num, self.patch_size)
        return x
