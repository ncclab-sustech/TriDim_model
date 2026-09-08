"""Sleep-EDF Database Expanded (cassette + telemetry)."""
from __future__ import annotations

import json
import logging
import re
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
from ..splits import apply_subject_disjoint_split
from ..validation import build_summary, qc_flag_mapping
from ..writer import create_zarr_store, row_groups_by_source, write_batch

SLEEPEDF_LABELS = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_MAP = {
    "sleep stage w": (0, "W"),
    "sleep stage 1": (1, "N1"),
    "sleep stage 2": (2, "N2"),
    "sleep stage 3": (3, "N3"),
    "sleep stage 4": (3, "N3"),  # mapped id N3, raw preserved separately
    "sleep stage r": (4, "REM"),
}
SEGMENT_SEC = 30.0
_PSG_RE = re.compile(r"^(?P<prefix>SC|ST)(?P<body>\d+)(?P<rest>.*)-PSG\.edf$", re.IGNORECASE)


def _person_id(stem: str) -> str:
    # Cassette SC4ssN* and telemetry ST7ssN*: keep person identity, not night.
    match = re.match(r"^SC4(\d{2})", stem, flags=re.IGNORECASE)
    if match:
        return f"SC{match.group(1)}"
    match = re.match(r"^ST7(\d{2})", stem, flags=re.IGNORECASE)
    if match:
        return f"ST{match.group(1)}"
    return stem[:5].upper()


def _pick_sleep_eeg_channels(ch_names: list[str]) -> tuple[list[str], list[str]]:
    keep_raw: list[str] = []
    keep_canon: list[str] = []
    for name in ch_names:
        upper = name.upper()
        if any(tok in upper for tok in ("EOG", "EMG", "RESP", "ECG", "EKG", "TEMP", "EVENT", "MARKER")):
            continue
        if upper.startswith("EEG") or "EEG" in upper:
            keep_raw.append(name)
            keep_canon.append(name.replace("EEG", "").strip() or name)
    return keep_raw, keep_canon


def _find_hypnogram(psg_path: Path) -> Path | None:
    stem = psg_path.stem.replace("-PSG", "")
    parent = psg_path.parent
    # Cassette/telemetry hypnograms share the subject/night prefix.
    candidates = sorted(parent.glob(f"{stem[:6]}*-Hypnogram.edf"))
    if not candidates:
        candidates = sorted(parent.glob(f"{stem[:5]}*-Hypnogram.edf"))
    return candidates[0] if candidates else None


def scan_sleepedf_plan(root: Path, smoke: bool = False) -> DatasetPlan:
    if not root.exists():
        raise FileNotFoundError(root)
    psg_files: list[Path] = []
    for sub in ("sleep-cassette", "sleep-telemetry"):
        folder = root / sub
        if folder.exists():
            psg_files.extend(sorted(folder.glob("*-PSG.edf")))
    if not psg_files:
        psg_files = sorted(root.rglob("*-PSG.edf"))
    if smoke:
        psg_files = psg_files[:2]
    if not psg_files:
        raise FileNotFoundError(f"No Sleep-EDF PSG files under {root}")

    rows: list[dict[str, Any]] = []
    channel_count_distribution: Counter[str] = Counter()

    for psg_path in psg_files:
        hyp_path = _find_hypnogram(psg_path)
        if hyp_path is None:
            logging.warning("SleepEDF missing hypnogram for %s", psg_path.name)
            continue
        person_id = _person_id(psg_path.stem)
        raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose=False)
        raw_names, channel_names = _pick_sleep_eeg_channels(list(raw.ch_names))
        if not raw_names:
            continue
        sfreq = float(raw.info["sfreq"])
        channel_count_distribution[str(len(channel_names))] += 1
        annotations = mne.read_annotations(str(hyp_path))

        for ann_idx, (onset, duration, desc) in enumerate(
            zip(annotations.onset, annotations.duration, annotations.description)
        ):
            key = str(desc).strip().lower()
            if key not in STAGE_MAP:
                continue
            label_id, __canon = STAGE_MAP[key]
            if float(duration) < SEGMENT_SEC - 1e-6:
                continue
            n_epochs = int(float(duration) // SEGMENT_SEC)
            for ep in range(n_epochs):
                start = float(onset) + float(ep) * SEGMENT_SEC
                end = start + SEGMENT_SEC
                rows.append(
                    {
                        "sample_id": f"SleepEDF_full:{person_id}:{psg_path.stem}:ann{ann_idx:04d}:ep{ep:04d}",
                        "dataset_key": "SleepEDF_full",
                        "subject_id": person_id,
                        "session_id": psg_path.stem,
                        "segment_id": f"{psg_path.stem}_ann{ann_idx:04d}_ep{ep:04d}",
                        "source_relpath": relpath(psg_path, root),
                        "source_format": "edf",
                        "segment_start_sec": start,
                        "segment_end_sec": end,
                        "label_id": label_id,
                        "label_raw": str(desc),
                        "channel_names_json": json.dumps(channel_names),
                        "raw_sfreq": sfreq,
                        "notch_hz": 50.0,
                        "qc_flags": 0,
                        "channel_count": len(channel_names),
                        "subset": "cassette" if "cassette" in str(psg_path).lower() else "telemetry",
                        "hypnogram_relpath": relpath(hyp_path, root),
                    }
                )

    rows.sort(
        key=lambda row: (
            row["source_relpath"],
            row["segment_start_sec"],
            row["segment_id"],
        )
    )
    apply_subject_disjoint_split(rows)
    c_max = max((int(row["channel_count"]) for row in rows), default=2)
    dataset_config = {
        "dataset_key": "SleepEDF_full",
        "source_root": str(root),
        "source_format": "edf",
        "segment_seconds": SEGMENT_SEC,
        "target_sampling_rate": TARGET_SFREQ,
        "signal_unit": "uV",
        "notch_hz": 50.0,
        "bandpass_low": 0.3,
        "bandpass_high": 35.0,
        "qc_flags": qc_flag_mapping(),
        "split_rule": "subject_disjoint_person_identity",
        "sample_order": "source-file/onset order; independent of split assignment",
        "label_note": "Raw hypnogram descriptions preserved; stage 4 maps to N3 id while label_raw keeps Sleep stage 4.",
        "includes": ["sleep-cassette", "sleep-telemetry"],
    }
    return DatasetPlan(
        dataset_key="SleepEDF_full",
        rows=rows,
        c_max=c_max,
        t=int(SEGMENT_SEC * TARGET_SFREQ),
        segment_seconds=SEGMENT_SEC,
        task_type="multiclass_classification",
        label_vocab={str(k): {"name": v} for k, v in SLEEPEDF_LABELS.items()},
        dataset_config=dataset_config,
        channel_count_distribution=dict(channel_count_distribution),
        expected_n_samples=len(rows),
    )


def write_sleepedf_zarr(root_path: Path, dataset_dir: Path, plan: DatasetPlan) -> dict[str, Any]:
    root = create_zarr_store(dataset_dir, plan)
    groups = row_groups_by_source(plan.rows)
    qc_counter: Counter[int] = Counter()
    label_counter: Counter[int] = Counter()
    start_time = time.time()

    for file_idx, (source_rel, indexed_rows) in enumerate(groups.items(), start=1):
        edf_path = root_path / source_rel
        logging.info("SleepEDF %d/%d processing %s", file_idx, len(groups), source_rel)
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        raw_names, channel_names = _pick_sleep_eeg_channels(list(raw.ch_names))
        raw.pick(raw_names)
        data_uv = raw.get_data() * 1e6
        raw_sfreq = float(raw.info["sfreq"])
        for __idx, row in indexed_rows:
            row["channel_names_json"] = json.dumps(channel_names)
            row["channel_count"] = len(channel_names)

        processed, __eff_high, file_qc = preprocess_continuous_uv(
            data_uv, raw_sfreq, notch_hz=50.0, bandpass_low=0.3, bandpass_high=35.0
        )
        indices_out: list[int] = []
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
                processed, row["segment_start_sec"], plan.t, TARGET_SFREQ
            )
            signal, channel_mask, bad_mask, final_qc = finalize_segment(
                segment,
                plan.c_max,
                plan.t,
                channel_names,
                file_qc | seg_qc,
                valid_samples,
                min_expected_channels=1,
            )
            row["qc_flags"] = int(final_qc)
            indices_out.append(idx)
            signals.append(signal)
            channel_masks.append(channel_mask)
            bad_masks.append(bad_mask)
            channel_counts_list.append(len(channel_names))
            labels.append(int(row["label_id"]))
            label_values.append(float(row["label_id"]))
            subject_ids.append(str(row["subject_id"]))
            qc_flags_list.append(int(final_qc))
            valid_time_list.append(int(valid_samples))
            qc_counter[int(final_qc)] += 1
            label_counter[int(row["label_id"])] += 1

        write_batch(
            root,
            indices_out,
            signals,
            channel_masks,
            bad_masks,
            channel_counts_list,
            labels,
            label_values,
            subject_ids,
            qc_flags_list,
            valid_time_list,
        )

    write_parquet_compat(pd.DataFrame(plan.rows), dataset_dir / "sample_index.parquet")
    write_json(dataset_dir / "label_vocab.json", plan.label_vocab)
    write_json(dataset_dir / "dataset_config.json", plan.dataset_config)
    summary = build_summary(plan, label_counter, qc_counter, elapsed_sec=time.time() - start_time)
    write_json(dataset_dir / "conversion_summary.json", summary)
    return summary
