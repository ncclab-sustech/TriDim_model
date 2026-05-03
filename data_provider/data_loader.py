import copy
import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import re
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (normalize_batch_ts,bandpass_filter_func)
import warnings
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from natsort import natsorted
try:
    import zarr
except Exception:
    zarr = None

warnings.filterwarnings("ignore")

class APAVALoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        self.split_mode = str(getattr(args, "split_mode", "label_order"))
        self.split_seed = int(getattr(args, "seed", getattr(args, "seed_start", 42)))
        self.train_ratio = float(getattr(args, "train_ratio", 0.4))
        self.val_ratio = float(getattr(args, "val_ratio", 0.3))
        if (
            self.train_ratio <= 0
            or self.val_ratio <= 0
            or self.train_ratio + self.val_ratio >= 1.0
        ):
            self.train_ratio, self.val_ratio = 0.4, 0.3

        # Keep legacy subject-level APAVA split by default.
        if self.split_mode == "segment_stratified_random":
            self.X, self.y = self.load_apava_segment_split(
                self.data_path, self.label_path, flag=flag
            )
        else:
            self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
                self.label_path, a=0.4, b=0.7
            )
            self.X, self.y = self.load_apava(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=256, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.4, b=0.7):
        """
        Build subject-level 4:3:3 split for APAVA.
        The split is stratified by class and keeps metadata order per class.
        """
        data_list = np.load(label_path)
        by_label = {}
        for row in data_list:
            lbl = int(row[0])
            sid = int(row[1])
            by_label.setdefault(lbl, [])
            if sid not in by_label[lbl]:
                by_label[lbl].append(sid)

        train_ids, val_ids, test_ids = [], [], []
        for lbl in sorted(by_label.keys()):
            members = by_label[lbl]
            n = len(members)
            t0 = int(a * n)
            t1 = int(b * n)
            train_ids.extend(members[:t0])
            val_ids.extend(members[t0:t1])
            test_ids.extend(members[t1:])

        return train_ids, val_ids, test_ids

    def load_apava(self, data_path, label_path, flag=None):
        """
        Loads APAVA data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = natsorted(list(os.listdir(data_path)))
        subject_label = np.load(label_path)

        if flag == "TRAIN":
            ids = set(self.train_ids)
        elif flag == "VAL":
            ids = set(self.val_ids)
        elif flag == "TEST":
            ids = set(self.test_ids)
        else:
            ids = set(int(v) for v in subject_label[:, 1].tolist())

        for j, filename in enumerate(filenames):
            trial_label = subject_label[j]
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            if int(trial_label[1]) in ids:
                for trial_feature in subject_feature:
                    feature_list.append(trial_feature)
                    label_list.append(int(trial_label[0]))

        X = np.asarray(feature_list, dtype=np.float32)
        y = np.asarray(label_list, dtype=np.int64)
        X, y = shuffle(X, y, random_state=42)
        return X, y

    def load_apava_segment_split(self, data_path, label_path, flag=None):
        """
        Non-cross-subject split for APAVA:
        collect all segments, then stratified 4:3:3 (or CLI ratios) by segment label.
        """
        feature_list = []
        label_list = []
        filenames = natsorted(list(os.listdir(data_path)))
        subject_label = np.load(label_path)

        for j, filename in enumerate(filenames):
            trial_label = subject_label[j]
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(int(trial_label[0]))

        X_all = np.asarray(feature_list, dtype=np.float32)
        y_all = np.asarray(label_list, dtype=np.int64)

        all_idx = np.arange(len(y_all))
        train_idx, rest_idx = train_test_split(
            all_idx,
            test_size=(1.0 - self.train_ratio),
            random_state=self.split_seed,
            stratify=y_all,
            shuffle=True,
        )
        val_portion = self.val_ratio / (1.0 - self.train_ratio)
        val_idx, test_idx = train_test_split(
            rest_idx,
            test_size=(1.0 - val_portion),
            random_state=self.split_seed,
            stratify=y_all[rest_idx],
            shuffle=True,
        )

        if flag == "TRAIN":
            idx = train_idx
        elif flag == "VAL":
            idx = val_idx
        elif flag == "TEST":
            idx = test_idx
        else:
            idx = all_idx

        X = X_all[idx]
        y = y_all[idx]
        X, y = shuffle(X, y, random_state=42)
        return X, y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class TDBRAINLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        train_ids = list(range(1, 18)) + list(
            range(29, 46)
        )  # specify patient ID for training, validation, and test set
        val_ids = [18, 19, 20, 21] + [46, 47, 48, 49]  # 8 patients, 4 positive 4 healthy
        test_ids = [22, 23, 24, 25] + [50, 51, 52, 53]  # 8 patients, 4 positive 4 healthy

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids

        self.X, self.y = self.load_tdbrain(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=256, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_tdbrain(self, data_path, label_path, flag=None):
        """
        Loads tdbrain data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if int(trial_label[1]) in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADHDLoader(Dataset):
    _label_map_cache = {}

    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.file_paths = natsorted(glob.glob(os.path.join(root_path, "sub_*.h5")))
        if len(self.file_paths) == 0:
            self.file_paths = natsorted(glob.glob(os.path.join(root_path, "*.h5")))
        self.zarr_path = None
        if len(self.file_paths) == 0:
            if str(root_path).lower().endswith(".zarr") and os.path.isdir(root_path):
                self.zarr_path = root_path
            else:
                zarr_candidates = natsorted(glob.glob(os.path.join(root_path, "*.zarr")))
                if len(zarr_candidates) > 0:
                    self.zarr_path = zarr_candidates[0]
        self.use_zarr = self.zarr_path is not None
        if len(self.file_paths) == 0 and not self.use_zarr:
            raise FileNotFoundError(f"No ADHD h5/zarr dataset found under: {root_path}")

        self.split_mode = str(getattr(args, "split_mode", "stratified_random"))
        self.split_seed = int(getattr(args, "seed", getattr(args, "seed_start", 42)))
        self.train_ratio = float(getattr(args, "train_ratio", 0.4))
        self.val_ratio = float(getattr(args, "val_ratio", 0.3))
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.train_ratio + self.val_ratio >= 1.0:
            self.train_ratio, self.val_ratio = 0.4, 0.3
        if self.use_zarr:
            self._load_zarr_arrays()
        if self.split_mode == "segment_stratified_random":
            self.train_ids, self.val_ids, self.test_ids = [], [], []
        else:
            if self.use_zarr:
                subject_to_label = self._build_subject_label_map_zarr()
            else:
                subject_to_label = self._build_subject_label_map(self.file_paths)
            if self.split_mode == "label_order":
                self.train_ids, self.val_ids, self.test_ids = self._split_subjects_by_label_order(
                    subject_to_label, train_r=self.train_ratio, val_r=self.val_ratio
                )
            else:
                self.train_ids, self.val_ids, self.test_ids = self._split_subjects_stratified_random(
                    subject_to_label, train_r=self.train_ratio, val_r=self.val_ratio, seed=self.split_seed
                )

        cache_key = os.path.abspath(self.root_path) + ("::zarr" if self.use_zarr else "::h5")
        if cache_key in ADHDLoader._label_map_cache:
            self.label_map = ADHDLoader._label_map_cache[cache_key]
        else:
            if self.use_zarr:
                unique_labels = self._collect_unique_labels_zarr()
            else:
                unique_labels = self._collect_unique_labels(self.file_paths)
            self.label_map = {int(lbl): i for i, lbl in enumerate(sorted(unique_labels))}
            ADHDLoader._label_map_cache[cache_key] = self.label_map

        if self.use_zarr:
            self.X, self.y = self.load_adhd_zarr(flag=flag)
        else:
            self.X, self.y = self.load_adhd(flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

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
        raise ValueError(f"Unexpected ADHD filename format: {file_path}")

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
                raise ValueError("Empty 'label' attr in ADHD eeg segment.")
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
            raise RuntimeError("No valid labels found in ADHD-style h5 dataset.")
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
        # Raw zarr shape is (N, C, T); convert to (N, T, C) for TeCh.
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
            raise RuntimeError("No valid ADHD subjects with labels were found.")
        return subject_to_label

    def _split_subjects_by_label_order(self, subject_to_label, train_r=0.4, val_r=0.3):
        by_label = {}
        for sid in sorted(subject_to_label.keys(), key=lambda x: str(x)):
            lbl = int(subject_to_label[sid])
            by_label.setdefault(lbl, [])
            by_label[lbl].append(sid)

        train_ids, val_ids, test_ids = [], [], []
        split_b = train_r + val_r
        for lbl in sorted(by_label.keys()):
            members = by_label[lbl]
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

        # Guard against accidental empty split when class count is very small.
        if len(val_ids) == 0 and len(train_ids) > 1:
            val_ids.append(train_ids.pop(-1))
        if len(test_ids) == 0 and len(train_ids) > 1:
            test_ids.append(train_ids.pop(-1))

        return train_ids, val_ids, test_ids

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

    def _load_all_segments(self):
        feature_list, label_list = [], []
        for file_path in self.file_paths:
            x_list, y_list = self._load_file_segments(file_path)
            feature_list.extend(x_list)
            label_list.extend(y_list)
        if len(feature_list) == 0:
            raise RuntimeError("No ADHD samples loaded from h5 files.")
        X = np.asarray(feature_list, dtype=np.float32)
        y = np.asarray(label_list, dtype=np.int64)
        return X, y

    def _split_segment_indices_stratified_random(self, labels, train_r=0.4, val_r=0.3, seed=42):
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        indices = np.arange(labels.shape[0], dtype=np.int64)
        if labels.shape[0] < 3:
            return indices, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        split_b = float(train_r) + float(val_r)
        rest_r = max(1e-8, 1.0 - float(train_r))
        val_from_rest = float(val_r) / rest_r
        val_from_rest = min(max(val_from_rest, 1e-8), 1.0 - 1e-8)

        strat_all = labels if len(np.unique(labels)) > 1 else None
        try:
            train_idx, rest_idx = train_test_split(
                indices,
                train_size=float(train_r),
                random_state=int(seed),
                shuffle=True,
                stratify=strat_all,
            )
            rest_labels = labels[rest_idx]
            strat_rest = rest_labels if len(np.unique(rest_labels)) > 1 else None
            val_idx, test_idx = train_test_split(
                rest_idx,
                train_size=val_from_rest,
                random_state=int(seed),
                shuffle=True,
                stratify=strat_rest,
            )
        except ValueError:
            rng = np.random.RandomState(int(seed))
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            n = len(shuffled)
            t0 = int(round(float(train_r) * n))
            t1 = int(round(split_b * n))
            t0 = min(max(1, t0), max(1, n - 2))
            t1 = min(max(t0 + 1, t1), max(t0 + 1, n - 1))
            train_idx = shuffled[:t0]
            val_idx = shuffled[t0:t1]
            test_idx = shuffled[t1:]

        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )

    def load_adhd_zarr(self, flag=None):
        X_all = self._zarr_X_all
        y_all = self._encode_labels(self._zarr_y_all_raw)
        sid_all = self._zarr_subject_ids

        if self.split_mode == "segment_stratified_random":
            train_idx, val_idx, test_idx = self._split_segment_indices_stratified_random(
                y_all, train_r=self.train_ratio, val_r=self.val_ratio, seed=self.split_seed
            )
            if flag == "TRAIN":
                idx = train_idx
            elif flag == "VAL":
                idx = val_idx
            elif flag == "TEST":
                idx = test_idx
            else:
                idx = np.arange(len(y_all), dtype=np.int64)
            X = X_all[idx]
            y = y_all[idx]
            X, y = shuffle(X, y, random_state=42)
            return X, y

        if flag == "TRAIN":
            target_ids = set(self.train_ids)
        elif flag == "VAL":
            target_ids = set(self.val_ids)
        elif flag == "TEST":
            target_ids = set(self.test_ids)
        else:
            target_ids = set(np.unique(sid_all).tolist())

        mask = np.asarray([sid in target_ids for sid in sid_all], dtype=bool)
        if not np.any(mask):
            raise RuntimeError(f"No zarr samples loaded for split: {flag}")
        X = X_all[mask]
        y = y_all[mask]
        X, y = shuffle(X, y, random_state=42)
        return X, y

    def load_adhd(self, flag=None):
        if self.split_mode == "segment_stratified_random":
            X_all, y_all = self._load_all_segments()
            y_all = self._encode_labels(y_all)
            train_idx, val_idx, test_idx = self._split_segment_indices_stratified_random(
                y_all, train_r=self.train_ratio, val_r=self.val_ratio, seed=self.split_seed
            )
            if flag == "TRAIN":
                X = X_all[train_idx]
                y = y_all[train_idx]
            elif flag == "VAL":
                X = X_all[val_idx]
                y = y_all[val_idx]
            elif flag == "TEST":
                X = X_all[test_idx]
                y = y_all[test_idx]
            else:
                X = X_all
                y = y_all
            X, y = shuffle(X, y, random_state=42)
            return X, y

        if flag == "TRAIN":
            target_ids = set(self.train_ids)
        elif flag == "VAL":
            target_ids = set(self.val_ids)
        elif flag == "TEST":
            target_ids = set(self.test_ids)
        else:
            target_ids = set(self._subject_id_from_path(p) for p in self.file_paths)

        feature_list, label_list = [], []
        for file_path in self.file_paths:
            sid = self._subject_id_from_path(file_path)
            if sid not in target_ids:
                continue
            x_list, y_list = self._load_file_segments(file_path)
            feature_list.extend(x_list)
            label_list.extend(y_list)

        if len(feature_list) == 0:
            raise RuntimeError(f"No ADHD samples loaded for split: {flag}")

        X = np.asarray(feature_list, dtype=np.float32)
        y = self._encode_labels(np.asarray(label_list, dtype=np.int64))
        X, y = shuffle(X, y, random_state=42)
        return X, y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class ADFTDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        
        if (
            self.train_ratio <= 0
            or self.val_ratio <= 0
            or self.train_ratio + self.val_ratio >= 1.0
        ):
            a, b = 0.4, 0.7
        else:
            a, b = self.train_ratio, self.train_ratio + self.val_ratio
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )
        self.X, self.y = self.load_adfd(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = bandpass_filter_func(self.X, fs=256, lowcut=0.5, highcut=45)
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.4, b=0.7):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        # Deduplicate subject IDs while preserving first-seen order.
        cn_list = list(dict.fromkeys(int(v) for v in data_list[np.where(data_list[:, 0] == 0)][:, 1]))
        ftd_list = list(dict.fromkeys(int(v) for v in data_list[np.where(data_list[:, 0] == 1)][:, 1]))
        ad_list = list(dict.fromkeys(int(v) for v in data_list[np.where(data_list[:, 0] == 2)][:, 1]))

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ftd_list[: int(a * len(ftd_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ftd_list[int(a * len(ftd_list)) : int(b * len(ftd_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ftd_list[int(b * len(ftd_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_adfd(self, data_path, label_path, flag=None):
        """
        Loads adfd or cnbpm data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = natsorted(list(os.listdir(data_path)))
        subject_label = np.load(label_path)

        if flag == "TRAIN":
            ids = set(self.train_ids)
        elif flag == "VAL":
            ids = set(self.val_ids)
        elif flag == "TEST":
            ids = set(self.test_ids)
        else:
            ids = set(int(v) for v in subject_label[:, 1].tolist())

        for j, filename in enumerate(filenames):
            trial_label = subject_label[j]
            path = data_path + filename
            subject_feature = np.load(path)
            if int(trial_label[1]) in ids:
                for trial_feature in subject_feature:
                    feature_list.append(trial_feature)
                    label_list.append(int(trial_label[0]))

        X = np.asarray(feature_list, dtype=np.float32)
        y = np.asarray(label_list, dtype=np.int64)
        X, y = shuffle(X, y, random_state=42)
        return X, y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.55, 0.7

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptb(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=250, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        my_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial infarction IDs

        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + my_list[int(a * len(my_list)) : int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)) :] + my_list[int(b * len(my_list)) :]

        return train_ids, val_ids, test_ids

    def load_ptb(self, data_path, label_path, flag=None):
        """
        Loads ptb data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if int(trial_label[1]) in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBXLLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptbxl(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=250, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        no_list = list(
            data_list[np.where(data_list[:, 0] == 0)][:, 1]
        )  # Normal ECG IDs
        mi_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial Infarction IDs
        sttc_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # ST/T Change IDs
        cd_list = list(
            data_list[np.where(data_list[:, 0] == 3)][:, 1]
        )  # Conduction Disturbance IDs
        hyp_list = list(
            data_list[np.where(data_list[:, 0] == 4)][:, 1]
        )  # Hypertrophy IDs

        train_ids = (
            no_list[: int(a * len(no_list))]
            + mi_list[: int(a * len(mi_list))]
            + sttc_list[: int(a * len(sttc_list))]
            + cd_list[: int(a * len(cd_list))]
            + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
            no_list[int(a * len(no_list)) : int(b * len(no_list))]
            + mi_list[int(a * len(mi_list)) : int(b * len(mi_list))]
            + sttc_list[int(a * len(sttc_list)) : int(b * len(sttc_list))]
            + cd_list[int(a * len(cd_list)) : int(b * len(cd_list))]
            + hyp_list[int(a * len(hyp_list)) : int(b * len(hyp_list))]
        )
        test_ids = (
            no_list[int(b * len(no_list)) :]
            + mi_list[int(b * len(mi_list)) :]
            + sttc_list[int(b * len(sttc_list)) :]
            + cd_list[int(b * len(cd_list)) :]
            + hyp_list[int(b * len(hyp_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_ptbxl(self, data_path, label_path, flag=None):
        """
        Loads ptb-xl data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if int(trial_label[1]) in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)



class FLAAPLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/feature.npy')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_flaap_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_flaap_dependent(self, data_path, label_path, flag=None):
        '''
        Loads fl-aap data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        X_train = np.load(data_path)
        y_train = np.load(label_path)
        # print(X_train.shape, y_train.shape)

        # 60 : 20 : 20
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train
        elif flag == 'VAL':
            return X_val, y_val
        elif flag == 'TEST':
            return X_test, y_test
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
            torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class UCIHARLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/feature.npy')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_har_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_har_dependent(self, data_path, label_path, flag=None):
        '''
        Loads fl-aap data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        X_train = np.load(data_path)
        y_train = np.load(label_path)
        # print(X_train.shape, y_train.shape)

        X_test = X_train[-2947:]
        y_test = y_train[-2947:]

        X_train, X_val, y_train, y_val = train_test_split(X_train[:-2947], y_train[:-2947], test_size=0.2, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train
        elif flag == 'VAL':
            return X_val, y_val
        elif flag == 'TEST':
            return X_test, y_test
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
            torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)
