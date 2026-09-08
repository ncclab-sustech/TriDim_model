"""AD65 (OpenNeuro ds004504) dataset: 3-class dementia classification from
resting-state eyes-closed EEG.

Source: OpenNeuro ds004504 (Miltiadous et al.).
- 19 scalp EEG channels in standard 10-20 layout, 500 Hz sampling rate.
- Task: task-eyesclosed, one continuous recording per subject.
- Mains frequency: 50 Hz.

Each subject is tagged in ``participants.tsv`` with ``Group`` in {A, F, C}:
    A  Alzheimer's Disease     -> label_id 2
    F  Frontotemporal Dementia -> label_id 1
    C  Healthy Control         -> label_id 0

We keep all three groups at preprocessing time (QC flags mark, never delete).
Downstream tasks can subset/remap labels (e.g. binary AD-vs-control).

Segmentation: 10-second non-overlapping windows (t_samples = 2000 at 200 Hz).
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from ..constants import TARGET_SFREQ
from ..dsp import extract_segment, finalize_segment, preprocess_continuous_uv
from ..io_utils import relpath, write_json, write_parquet_compat
from ..plan import DatasetPlan
from ..validation import build_summary, qc_flag_mapping
from ..writer import create_zarr_store, row_groups_by_source, write_batch

logger = logging.getLogger(__name__)

DATASET_KEY = "AD65"

AD65_CHANNELS: list[str] = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
C_MAX = len(AD65_CHANNELS)  # 19

GROUP_TO_LABEL_ID: dict[str, int] = {"C": 0, "F": 1, "A": 2}
AD65_LABEL_NAMES: dict[int, str] = {
    0: "control",
    1: "frontotemporal_dementia",
    2: "alzheimer",
}

SEGMENT_SEC = 10.0
T_SAMPLES = int(SEGMENT_SEC * TARGET_SFREQ)  # 2000
RAW_SFREQ = 500.0
NOTCH_HZ = 50.0

SMOKE_SUBJECTS = {"sub-001", "sub-037"}


def _load_participants(root: Path) -> dict[str, str]:
    tsv_path = root / "participants.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")
    if "participant_id" not in df.columns or "Group" not in df.columns:
        raise ValueError(
            f"participants.tsv missing required columns: {df.columns.tolist()}"
        )
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        pid = str(row["participant_id"]).strip()
        group = str(row["Group"]).strip()
        if group not in GROUP_TO_LABEL_ID:
            raise ValueError(f"Unknown Group '{group}' for {pid} in {tsv_path}")
        mapping[pid] = group
    return mapping


def _subject_id_from_path(path: Path) -> str:
    return path.stem.split("_")[0]


def _find_set_files(root: Path, smoke: bool = False) -> list[Path]:
    files = sorted(
        p for p in root.glob("sub-*_task-eyesclosed_eeg.set")
        if "@eaDir" not in p.parts
    )
    if smoke:
        files = [p for p in files if _subject_id_from_path(p) in SMOKE_SUBJECTS]
    return files


def scan_ad65_plan(root: Path, smoke: bool = False) -> DatasetPlan:
    if not root.exists():
        raise FileNotFoundError(root)

    participants = _load_participants(root)
    set_files = _find_set_files(root, smoke=smoke)
    if not set_files:
        raise RuntimeError(f"No sub-*_task-eyesclosed_eeg.set files under {root}")

    rows: list[dict[str, Any]] = []
    channel_counts: Counter[int] = Counter()
    group_counter: Counter[str] = Counter()
    skipped_files = 0

    for set_path in set_files:
        subject_id = _subject_id_from_path(set_path)
        if subject_id not in participants:
            logger.warning(
                "Subject %s not listed in participants.tsv; skipping", subject_id,
            )
            skipped_files += 1
            continue

        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
        except Exception as exc:
            logger.warning("Cannot read %s: %s", set_path, exc)
            skipped_files += 1
            continue

        raw_sfreq = float(raw.info["sfreq"])
        duration_sec = float(raw.n_times) / raw_sfreq
        ch_names = list(raw.ch_names)
        channel_counts[len(ch_names)] += 1

        group = participants[subject_id]
        label_id = GROUP_TO_LABEL_ID[group]
        group_counter[group] += 1

        n_segments = int(duration_sec // SEGMENT_SEC)
        if n_segments <= 0:
            logger.warning(
                "%s is shorter than one %.0fs segment (%.2fs); skipping",
                set_path.name, SEGMENT_SEC, duration_sec,
            )
            skipped_files += 1
            continue

        source_rel = relpath(set_path, root)
        session_id = set_path.stem  # sub-XXX_task-eyesclosed_eeg

        for seg_idx in range(n_segments):
            start_sec = float(seg_idx * SEGMENT_SEC)
            end_sec = start_sec + SEGMENT_SEC
            segment_id = f"{session_id}_seg{seg_idx:04d}"
            rows.append({
                "sample_id": f"{DATASET_KEY}:{subject_id}:{session_id}:{segment_id}",
                "dataset_key": DATASET_KEY,
                "subject_id": subject_id,
                "session_id": session_id,
                "trial_id": "",
                "segment_id": segment_id,
                "source_relpath": source_rel,
                "source_format": "set",
                "raw_inner_path": "",
                "segment_start_sec": start_sec,
                "segment_end_sec": end_sec,
                "event_start_sec": np.nan,
                "event_end_sec": np.nan,
                "event_id": "",
                "label_id": label_id,
                "label_raw": group,
                "label_source": "participants.tsv:Group",
                "channel_names_json": json.dumps(ch_names, ensure_ascii=False),
                "montage_name": "standard_1020_19ch",
                "raw_sfreq": raw_sfreq,
                "notch_hz": NOTCH_HZ,
                "bandpass_high_effective": 75.0,
                "split": "unassigned",
                "segment_kind": "regular",
                "qc_flags": 0,
                "csbrain_split_hint": "",
                "source_file_name": set_path.name,
                "task_code": "eyesclosed",
                "group": group,
                "segment_index": seg_idx,
                "channel_count": len(ch_names),
            })

    if not rows:
        raise RuntimeError(f"No valid AD65 segments built from {root}")

    rows.sort(key=lambda r: (r["subject_id"], r["session_id"], r["segment_index"]))

    logger.info(
        "AD65 scan: %d segments from %d subjects (A=%d, F=%d, C=%d); %d files skipped",
        len(rows), len({r["subject_id"] for r in rows}),
        group_counter.get("A", 0), group_counter.get("F", 0), group_counter.get("C", 0),
        skipped_files,
    )

    label_vocab = {
        str(label_id): {
            "name": AD65_LABEL_NAMES[label_id],
            "raw_aliases": [group_code],
        }
        for group_code, label_id in GROUP_TO_LABEL_ID.items()
    }
    dataset_config = {
        "dataset_key": DATASET_KEY,
        "source_root": "<provided-at-runtime>",
        "source_format": "set",
        "release": "ds004504-1.0.8",
        "task": "task-eyesclosed",
        "segment_seconds": SEGMENT_SEC,
        "segment_stride_seconds": SEGMENT_SEC,
        "target_sampling_rate": TARGET_SFREQ,
        "signal_unit": "uV",
        "notch_hz": NOTCH_HZ,
        "bandpass_low": 0.3,
        "bandpass_high": 75.0,
        "channel_rule": (
            "keep EEGLAB channels in source order; expected 19 channels of the "
            "10-20 montage (Fp1/Fp2/F3/F4/C3/C4/P3/P4/O1/O2/F7/F8/T3/T4/T5/T6/Fz/Cz/Pz)"
        ),
        "channel_names": AD65_CHANNELS,
        "label_rule": (
            "participants.tsv Group -> label_id: C->0, F->1, A->2. "
            "All three groups preserved; downstream tasks may subset (e.g. C vs A)"
        ),
        "segment_rule": f"non-overlapping {int(SEGMENT_SEC)}s windows starting at t=0",
        "split_rule": "split=unassigned; downstream splits assigned elsewhere",
        "hard_invalid_rule": "skip unreadable .set or subjects without participants.tsv entry",
        "qc_flags": qc_flag_mapping(),
        "skipped_file_count": skipped_files,
        "group_counts": {k: int(v) for k, v in sorted(group_counter.items())},
    }
    return DatasetPlan(
        dataset_key=DATASET_KEY,
        rows=rows,
        c_max=C_MAX,
        t=T_SAMPLES,
        segment_seconds=SEGMENT_SEC,
        task_type="multiclass_classification",
        label_vocab=label_vocab,
        dataset_config=dataset_config,
        channel_count_distribution={
            str(k): v for k, v in sorted(channel_counts.items())
        },
        expected_n_samples=len(rows),
    )


def write_ad65_zarr(
    root_path: Path, dataset_dir: Path, plan: DatasetPlan,
) -> dict[str, Any]:
    zarr_root = create_zarr_store(dataset_dir, plan)
    groups = row_groups_by_source(plan.rows)
    qc_counter: Counter[int] = Counter()
    label_counter: Counter[int] = Counter()
    start_time = time.time()

    for file_idx, (source_rel, indexed_rows) in enumerate(groups.items(), start=1):
        set_path = root_path / source_rel
        logger.info(
            "AD65 %d/%d processing %s (%d segments)",
            file_idx, len(groups), source_rel, len(indexed_rows),
        )

        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
        source_channels = list(raw.ch_names)
        data_uv = raw.get_data(units="uV")

        first_row = indexed_rows[0][1]
        raw_sfreq = float(first_row["raw_sfreq"])
        notch_hz = float(first_row["notch_hz"])

        processed, effective_high, file_qc = preprocess_continuous_uv(
            data_uv, raw_sfreq=raw_sfreq, notch_hz=notch_hz,
        )

        indices: list[int] = []
        signals: list[np.ndarray] = []
        channel_masks: list[np.ndarray] = []
        bad_masks: list[np.ndarray] = []
        channel_counts_list: list[int] = []
        labels: list[int] = []
        label_values: list[float] = []
        subject_ids: list[str] = []
        qc_flags_list: list[int] = []
        valid_time_list: list[int] = []

        for idx, row in indexed_rows:
            segment, valid_samples, seg_qc = extract_segment(
                processed,
                start_sec=float(row["segment_start_sec"]),
                t_samples=plan.t,
            )
            signal, ch_mask, bad_mask, qc_flags = finalize_segment(
                segment=segment,
                c_max=plan.c_max,
                t_samples=plan.t,
                channel_names=source_channels,
                inherited_qc=int(row["qc_flags"]) | file_qc | seg_qc,
                valid_time_samples=valid_samples,
                min_expected_channels=C_MAX,
            )
            row["qc_flags"] = int(qc_flags)
            row["valid_time_samples"] = int(valid_samples)
            row["bandpass_high_effective"] = float(effective_high)

            indices.append(idx)
            signals.append(signal)
            channel_masks.append(ch_mask)
            bad_masks.append(bad_mask)
            channel_counts_list.append(len(source_channels))
            labels.append(int(row["label_id"]))
            label_values.append(np.nan)
            subject_ids.append(str(row["subject_id"]))
            qc_flags_list.append(int(qc_flags))
            valid_time_list.append(int(valid_samples))
            qc_counter[int(qc_flags)] += 1
            label_counter[int(row["label_id"])] += 1

        write_batch(
            zarr_root, indices, signals, channel_masks, bad_masks,
            channel_counts_list, labels, label_values, subject_ids,
            qc_flags_list, valid_time_list,
        )
        del raw, data_uv, processed

    index_df = pd.DataFrame(plan.rows)
    write_parquet_compat(index_df, dataset_dir / "sample_index.parquet")
    write_json(dataset_dir / "label_vocab.json", plan.label_vocab)
    write_json(dataset_dir / "dataset_config.json", plan.dataset_config)
    summary = build_summary(
        plan, label_counter, qc_counter, elapsed_sec=time.time() - start_time,
    )
    write_json(dataset_dir / "conversion_summary.json", summary)
    return summary
