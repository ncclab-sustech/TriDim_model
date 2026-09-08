import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import re
import h5py
import torch
from torch.utils.data import Dataset
from data_provider.uea import normalize_batch_ts
import warnings
import hashlib
from sklearn.utils import shuffle
from natsort import natsorted
try:
    import zarr
except Exception:
    zarr = None

warnings.filterwarnings("ignore")


def _split_subjects_stratified_random(subject_to_label, train_r, val_r, seed):
    """Deterministically split subject IDs within each class."""
    rng = np.random.RandomState(int(seed))
    by_label = {}
    for sid, label in subject_to_label.items():
        by_label.setdefault(int(label), []).append(int(sid))

    train_ids, val_ids, test_ids = [], [], []
    split_b = float(train_r) + float(val_r)
    for label in sorted(by_label):
        members = sorted(by_label[label])
        rng.shuffle(members)
        n = len(members)
        if n < 3:
            raise ValueError(
                f"Class {label} has only {n} subjects; cannot create train/val/test splits"
            )
        train_end = min(max(1, int(round(float(train_r) * n))), n - 2)
        val_end = min(max(train_end + 1, int(round(split_b * n))), n - 1)
        train_ids.extend(members[:train_end])
        val_ids.extend(members[train_end:val_end])
        test_ids.extend(members[val_end:])
    return train_ids, val_ids, test_ids


def _load_external_subject_split(manifest_spec, seed, expected_ids):
    if not manifest_spec:
        raise ValueError("split_mode=external_manifest requires external_split_manifest")
    manifest_path = str(manifest_spec).format(seed=int(seed))
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"External subject split manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    splits = manifest.get("splits", {})
    result = {}
    for name in ("train", "val", "test"):
        values = [int(value) for value in splits.get(name, [])]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Invalid or duplicate subject IDs in {name}: {manifest_path}")
        result[name] = values

    split_sets = {name: set(values) for name, values in result.items()}
    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        raise ValueError(f"External subject split contains overlap: {manifest_path}")
    covered = set().union(*split_sets.values())
    if covered != set(int(value) for value in expected_ids):
        raise ValueError(f"External subject split does not exactly cover the dataset: {manifest_path}")

    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    print(
        "[subject_split] "
        f"manifest={manifest_path} sha256={fingerprint} "
        f"train/val/test={len(result['train'])}/{len(result['val'])}/{len(result['test'])}"
    )
    return result["train"], result["val"], result["test"]


class EEGDatasetLoader(Dataset):
    _label_map_cache = {}

    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.flag = str(flag).upper() if flag is not None else None
        if self.flag not in {None, "TRAIN", "VAL", "TEST"}:
            raise ValueError(f"Unsupported data split flag: {flag!r}")

        self.split_mode = str(getattr(args, "split_mode", "stratified_random"))
        if self.split_mode not in {"stratified_random", "external_manifest"}:
            raise ValueError(
                "Public release supports only subject-aware stratified_random "
                "or immutable external_manifest splits"
            )
        self.split_seed = int(getattr(args, "seed", getattr(args, "seed_start", 42)))
        self.train_ratio = float(getattr(args, "train_ratio", 0.8))
        self.val_ratio = float(getattr(args, "val_ratio", 0.1))
        if (
            self.train_ratio <= 0
            or self.val_ratio <= 0
            or self.train_ratio + self.val_ratio >= 1.0
        ):
            raise ValueError("train_ratio and val_ratio must be positive and sum to < 1")

        self.file_paths = natsorted(glob.glob(os.path.join(root_path, "sub_*.h5")))
        if not self.file_paths:
            self.file_paths = natsorted(glob.glob(os.path.join(root_path, "*.h5")))

        self.zarr_path = None
        if not self.file_paths:
            if str(root_path).lower().endswith(".zarr") and os.path.isdir(root_path):
                self.zarr_path = root_path
            else:
                candidates = natsorted(glob.glob(os.path.join(root_path, "*.zarr")))
                if candidates:
                    self.zarr_path = candidates[0]
        self.use_zarr = self.zarr_path is not None
        if not self.file_paths and not self.use_zarr:
            raise FileNotFoundError(f"No EEG HDF5/Zarr dataset found under: {root_path}")

        if self.use_zarr:
            self._load_zarr_arrays()

        manifest_spec = getattr(args, "external_split_manifest", None)
        self._external_split_indices = None
        if self.split_mode == "external_manifest":
            self._external_split_indices = self._load_external_zarr_split_indices(
                manifest_spec
            )
            if self._external_split_indices is None:
                raise ValueError(
                    "split_mode=external_manifest requires external_split_manifest"
                )
            self.train_ids, self.val_ids, self.test_ids = [], [], []
        else:
            if manifest_spec:
                raise ValueError(
                    "external_split_manifest requires split_mode=external_manifest"
                )
            subject_to_label = (
                self._build_subject_label_map_zarr()
                if self.use_zarr
                else self._build_subject_label_map(self.file_paths)
            )
            self.train_ids, self.val_ids, self.test_ids = (
                self._split_subjects_stratified_random(
                    subject_to_label,
                    train_r=self.train_ratio,
                    val_r=self.val_ratio,
                    seed=self.split_seed,
                )
            )

        cache_key = os.path.abspath(self.root_path) + (
            "::zarr" if self.use_zarr else "::h5"
        )
        if cache_key not in EEGDatasetLoader._label_map_cache:
            unique_labels = (
                self._collect_unique_labels_zarr()
                if self.use_zarr
                else self._collect_unique_labels(self.file_paths)
            )
            EEGDatasetLoader._label_map_cache[cache_key] = {
                int(label): index
                for index, label in enumerate(sorted(unique_labels))
            }
        self.label_map = EEGDatasetLoader._label_map_cache[cache_key]

        if self.use_zarr:
            self.X, self.y = self.load_zarr_split()
        else:
            self.X, self.y = self.load_h5_split()
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def _load_external_zarr_split_indices(self, manifest_spec):
        """Load a fixed train/val/test index manifest for an identical zarr export.

        ``manifest_spec`` may include ``{seed}``, which is resolved with the
        experiment seed.  This enables exact replay of an external benchmark
        split while retaining the model's usual training code.
        """
        if not manifest_spec:
            return None
        if not self.use_zarr:
            raise ValueError("external_split_manifest is supported only for zarr datasets")

        manifest_path = str(manifest_spec).format(seed=self.split_seed)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"External split manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        splits = manifest.get("splits", {})
        required = ("train", "val", "test")
        if any(name not in splits for name in required):
            raise KeyError(
                f"External split manifest must contain {required}; got {sorted(splits)}"
            )

        n_samples = int(self._zarr_X_all.shape[0])
        result = {}
        for name in required:
            indices = np.asarray(splits[name], dtype=np.int64).reshape(-1)
            if indices.size == 0:
                raise ValueError(f"External split '{name}' is empty in {manifest_path}")
            if indices.min() < 0 or indices.max() >= n_samples:
                raise IndexError(
                    f"External split '{name}' contains indices outside [0, {n_samples})"
                )
            if np.unique(indices).size != indices.size:
                raise ValueError(f"External split '{name}' contains duplicate indices")
            result[name] = indices

        if (
            np.intersect1d(result["train"], result["val"]).size
            or np.intersect1d(result["train"], result["test"]).size
            or np.intersect1d(result["val"], result["test"]).size
        ):
            raise ValueError("External train/val/test indices overlap")

        print(
            "[dataset_cfg] fixed external zarr split: "
            f"{manifest_path}; train/val/test="
            f"{len(result['train'])}/{len(result['val'])}/{len(result['test'])}"
        )
        return result

    def _subject_id_from_path(self, file_path):
        name = os.path.basename(file_path)
        m = re.search(r"sub_(\d+)\.h5$", name)
        if m is not None:
            return int(m.group(1))
        # Support names like sub-f1.h5 / sub-m2.h5
        m = re.search(r"sub-([A-Za-z]+)(\d+)\.h5$", name)
        if m is not None:
            prefix = m.group(1).lower()
            num = int(m.group(2))
            # Keep deterministic and disjoint spaces by prefix group.
            if prefix.startswith("f"):
                return 100000 + num
            if prefix.startswith("m"):
                return 200000 + num
            return 300000 + num
        # Support names like sub-1.h5
        m = re.search(r"sub-(\d+)\.h5$", name)
        if m is not None:
            return int(m.group(1))
        # Support names like sub-001_task-eyesclosed_eeg.h5 (AD65 style)
        m = re.search(r"sub-(\d+)", name)
        if m is not None:
            return int(m.group(1))
        # Support names like S001.h5 (Physionet_MI style)
        m = re.match(r"S(\d+)\.h5$", name, re.IGNORECASE)
        if m is not None:
            return int(m.group(1))
        # FACED_new style: sub000.h5 (no separator between "sub" and digits)
        m = re.match(r"sub(\d+)\.h5$", name)
        if m is not None:
            return int(m.group(1))
        # sleep-cassette-200hz style: sub_SC4001E0.h5 / sub_SC4012E0.h5
        m = re.match(r"sub_SC(\d+)\w*\.h5$", name)
        if m is not None:
            return 500000 + int(m.group(1))
        # BCIC2A style: A01T.h5 (training) / A01E.h5 (evaluation). Treat T/E as distinct subjects.
        # m = re.match(r"A(\d+)([ET])\.h5$", name)
        # if m is not None:
        #     num = int(m.group(1))
        #     return (600000 if m.group(2) == "T" else 700000) + num
        m = re.match(r"A(\d+)([ET])\.h5$", name)
        if m is not None:
            num = int(m.group(1))
            return 600000 + num
        # MDD style: "H S1 EC.h5" (healthy) / "MDD S1 EC.h5" (patient) / "6921143_H S15 EO.h5".
        # Group prefix disambiguates healthy vs patient; condition (EC/EO/TASK) merges into same subject.
        m = re.search(r"(?:^|[_\s])([A-Za-z]+)\s+S(\d+)\b", name)
        if m is not None:
            prefix = m.group(1).upper()
            num = int(m.group(2))
            if prefix == "H":
                return 800000 + num
            if prefix == "MDD":
                return 900000 + num
            return 950000 + num
        # Fallback for files like "10_20151125_noon.h5".
        m = re.match(r"(\d+)", name)
        if m is not None:
            return int(m.group(1))
        raise ValueError(f"Unexpected EEG filename format: {file_path}")

    def _extract_label(self, eeg_dataset, seg_group=None):
        raw_label = None
        if "label" in eeg_dataset.attrs:
            raw_label = eeg_dataset.attrs["label"]
        elif seg_group is not None and "label" in seg_group.attrs:
            raw_label = seg_group.attrs["label"]
        elif seg_group is not None and "label" in seg_group:
            raw_label = np.asarray(seg_group["label"])
        else:
            raise KeyError("Missing 'label' in eeg attrs and segment attrs/datasets.")
        if isinstance(raw_label, np.ndarray):
            if raw_label.size == 0:
                raise ValueError("Empty 'label' attr in EEG eeg segment.")
            raw_label = raw_label.reshape(-1)[0]
        return int(raw_label)

    def _collect_unique_labels(self, file_paths):
        labels = set()
        for file_path in file_paths:
            try:
                with h5py.File(file_path, "r") as f:
                    for trial_key in natsorted(list(f.keys())):
                        trial_group = f[trial_key]
                        if not isinstance(trial_group, h5py.Group):
                            continue
                        for seg_key in natsorted(list(trial_group.keys())):
                            seg_group = trial_group[seg_key]
                            if not isinstance(seg_group, h5py.Group):
                                continue
                            if "eeg" not in seg_group:
                                continue
                            eeg_ds = seg_group["eeg"]
                            labels.add(self._extract_label(eeg_ds, seg_group=seg_group))
            except OSError as e:
                warnings.warn(f"Skip unreadable h5 file while collecting labels: {file_path} ({e})")
        if not labels:
            raise RuntimeError("No valid labels found in EEG-style h5 dataset.")
        return labels

    def _load_zarr_arrays(self):
        if zarr is None:
            raise ImportError(
                "zarr package is required for .zarr datasets. "
                "Install with: python -m pip install zarr"
            )
        group = zarr.open_group(self.zarr_path, mode="r")
        required = ("signals", "labels")
        for key in required:
            if key not in group:
                raise KeyError(f"Missing '{key}' in zarr dataset: {self.zarr_path}")

        X = np.asarray(group["signals"], dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected signals ndim=3, got {X.ndim} in {self.zarr_path}")
        # Raw zarr shape is (N, C, T); convert to (N, T, C) for TriDim.
        X = np.transpose(X, (0, 2, 1))
        y = np.asarray(group["labels"], dtype=np.int64).reshape(-1)
        n = int(X.shape[0])
        if y.shape[0] != n:
            raise ValueError(
                f"signals/labels length mismatch: {n} vs {y.shape[0]} in {self.zarr_path}"
            )

        sid_source = "zarr:subject_ids"
        if "subject_ids" in group:
            sids = np.asarray(group["subject_ids"]).reshape(-1)
            sids = np.asarray([str(v) for v in sids], dtype=object)
            if sids.shape[0] != n:
                raise ValueError(
                    f"signals/subject_ids length mismatch: {n} vs {sids.shape[0]} in {self.zarr_path}"
                )
        else:
            # Newer downstream_preprocess exports may omit `subject_ids` from zarr
            # while preserving it in sample_index.parquet.
            sid_source = "sample_index.parquet"
            sids = self._load_subject_ids_from_sample_index(expected_len=n)


        self._zarr_subject_id_source = sid_source

        self._zarr_X_all = X
        self._zarr_y_all_raw = y
        self._zarr_subject_ids = sids

    def _load_subject_ids_from_sample_index(self, expected_len: int):
        parent_dir = str(Path(self.zarr_path).resolve().parent)
        sample_index_path = os.path.join(parent_dir, "sample_index.parquet")
        if not os.path.isfile(sample_index_path):
            raise KeyError(
                "Missing 'subject_ids' in zarr and no sample_index.parquet found at "
                f"{sample_index_path}"
            )

        candidate_cols = [
            "subject_id",
            "subject_ids",
            "subject",
            "sub_id",
            "participant_id",
        ]
        try:
            df = pd.read_parquet(sample_index_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read sample_index.parquet for subject fallback: {sample_index_path}"
            ) from exc

        selected_col = None
        for col in candidate_cols:
            if col in df.columns:
                selected_col = col
                break
        if selected_col is None:
            raise KeyError(
                "sample_index.parquet has no subject id column. "
                f"Checked: {candidate_cols}. File: {sample_index_path}"
            )

        sids = df[selected_col].astype(str).to_numpy(dtype=object).reshape(-1)
        if sids.shape[0] != int(expected_len):
            raise ValueError(
                "sample_index subject length mismatch with zarr signals: "
                f"{sids.shape[0]} vs {expected_len} ({sample_index_path})"
            )
        return sids

    def _collect_unique_labels_zarr(self):
        if not hasattr(self, "_zarr_y_all_raw"):
            self._load_zarr_arrays()
        return set(int(v) for v in np.unique(self._zarr_y_all_raw).tolist())

    def _build_subject_label_map_zarr(self):
        if not hasattr(self, "_zarr_subject_ids"):
            self._load_zarr_arrays()
        subject_to_label = {}
        unique_subjects = np.unique(self._zarr_subject_ids)
        for sid in unique_subjects.tolist():
            mask = self._zarr_subject_ids == sid
            labels = self._zarr_y_all_raw[mask]
            if labels.size == 0:
                continue
            vals, counts = np.unique(labels, return_counts=True)
            subject_to_label[sid] = int(vals[np.argmax(counts)])
        if len(subject_to_label) == 0:
            raise RuntimeError("No valid subjects with labels were found in zarr dataset.")
        return subject_to_label

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        encoded = []
        for lbl in y.tolist():
            key = int(lbl)
            if key not in self.label_map:
                raise KeyError(f"Unexpected label {key} not present in global label map.")
            encoded.append(self.label_map[key])
        return np.asarray(encoded, dtype=np.int64)

    def _load_file_segments(self, file_path):
        features, labels = [], []
        try:
            with h5py.File(file_path, "r") as f:
                for trial_key in natsorted(list(f.keys())):
                    trial_group = f[trial_key]
                    if not isinstance(trial_group, h5py.Group):
                        continue
                    for seg_key in natsorted(list(trial_group.keys())):
                        seg_group = trial_group[seg_key]
                        if not isinstance(seg_group, h5py.Group):
                            continue
                        if "eeg" not in seg_group:
                            continue
                        eeg_ds = seg_group["eeg"]
                        x = np.asarray(eeg_ds, dtype=np.float32)
                        if x.ndim != 2:
                            continue
                        # Expected raw shape is (C, T); convert to (T, C) to match TeCh loaders.
                        if x.shape[0] < x.shape[1]:
                            x = x.T
                        y = self._extract_label(eeg_ds, seg_group=seg_group)
                        features.append(x)
                        labels.append(y)
        except OSError as e:
            warnings.warn(f"Skip unreadable h5 file: {file_path} ({e})")
        return features, labels

    def _build_subject_label_map(self, file_paths):
        subject_to_label = {}
        for file_path in file_paths:
            sid = self._subject_id_from_path(file_path)
            _, labels = self._load_file_segments(file_path)
            if len(labels) == 0:
                continue
            # Use majority segment label as subject label.
            unique_vals, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
            subject_to_label[sid] = int(unique_vals[np.argmax(counts)])
        if len(subject_to_label) == 0:
            raise RuntimeError("No valid EEG subjects with labels were found.")
        return subject_to_label


    def _split_subjects_stratified_random(self, subject_to_label, train_r=0.4, val_r=0.3, seed=42):
        rng = np.random.RandomState(int(seed))
        by_label = {}
        for sid, lbl in subject_to_label.items():
            by_label.setdefault(int(lbl), [])
            by_label[int(lbl)].append(sid)

        train_ids, val_ids, test_ids = [], [], []
        split_b = float(train_r) + float(val_r)

        for lbl in sorted(by_label.keys()):
            members = list(by_label[lbl])
            rng.shuffle(members)
            n = len(members)
            if n == 1:
                t0, t1 = 1, 1
            elif n == 2:
                t0, t1 = 1, 2
            else:
                t0 = max(1, int(round(train_r * n)))
                t0 = min(t0, n - 2)
                t1 = int(round(split_b * n))
                t1 = max(t0 + 1, t1)
                t1 = min(t1, n - 1)
            train_ids.extend(members[:t0])
            val_ids.extend(members[t0:t1])
            test_ids.extend(members[t1:])

        if len(val_ids) == 0 and len(train_ids) > 1:
            val_ids.append(train_ids.pop(-1))
        if len(test_ids) == 0 and len(train_ids) > 1:
            test_ids.append(train_ids.pop(-1))

        return train_ids, val_ids, test_ids



    def load_zarr_split(self):
        X_all = self._zarr_X_all
        y_all = self._encode_labels(self._zarr_y_all_raw)
        subject_ids = self._zarr_subject_ids

        if self._external_split_indices is not None:
            split_name = {
                "TRAIN": "train",
                "VAL": "val",
                "TEST": "test",
            }.get(self.flag)
            indices = (
                np.arange(len(y_all), dtype=np.int64)
                if split_name is None
                else self._external_split_indices[split_name]
            )
            X, y = X_all[indices], y_all[indices]
            if (
                split_name == "train"
                and os.environ.get("SHUFFLE_TRAIN_LABELS", "0") == "1"
            ):
                shuffle_seed = int(
                    os.environ.get("SHUFFLE_TRAIN_LABELS_SEED", "42")
                )
                rng = np.random.RandomState(shuffle_seed)
                y = y[rng.permutation(len(y))]
                print(
                    "[SHUFFLE_TRAIN_LABELS] train labels permuted with "
                    f"seed {shuffle_seed}; validation/test labels unchanged"
                )
        else:
            if self.flag == "TRAIN":
                target_ids = set(self.train_ids)
            elif self.flag == "VAL":
                target_ids = set(self.val_ids)
            elif self.flag == "TEST":
                target_ids = set(self.test_ids)
            else:
                target_ids = set(np.unique(subject_ids).tolist())
            mask = np.asarray(
                [subject_id in target_ids for subject_id in subject_ids],
                dtype=bool,
            )
            if not np.any(mask):
                raise RuntimeError(f"No Zarr EEG samples loaded for split: {self.flag}")
            X, y = X_all[mask], y_all[mask]

        return shuffle(X, y, random_state=42)

    def load_h5_split(self):
        if self.flag == "TRAIN":
            target_ids = set(self.train_ids)
        elif self.flag == "VAL":
            target_ids = set(self.val_ids)
        elif self.flag == "TEST":
            target_ids = set(self.test_ids)
        else:
            target_ids = {
                self._subject_id_from_path(path) for path in self.file_paths
            }

        features, labels = [], []
        for file_path in self.file_paths:
            subject_id = self._subject_id_from_path(file_path)
            if subject_id not in target_ids:
                continue
            file_features, file_labels = self._load_file_segments(file_path)
            features.extend(file_features)
            labels.extend(file_labels)

        if not features:
            raise RuntimeError(f"No HDF5 EEG samples loaded for split: {self.flag}")
        X = np.asarray(features, dtype=np.float32)
        y = self._encode_labels(np.asarray(labels, dtype=np.int64))
        return shuffle(X, y, random_state=42)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)
