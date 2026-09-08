"""BCI Competition IV 2a dataset: 4-class motor imagery."""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..constants import QC_SOURCE_ARTIFACT, TARGET_SFREQ
from ..dsp import extract_segment, finalize_segment, preprocess_continuous_uv
from ..io_utils import relpath, write_json, write_parquet_compat
from ..plan import DatasetPlan
from ..validation import build_summary, qc_flag_mapping
from ..writer import create_zarr_store, row_groups_by_source, write_batch

BCIC2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

BCIC2A_LABEL_NAMES = {
    0: "left_hand",
    1: "right_hand",
    2: "feet",
    3: "tongue",
}


def bcic2a_a_files(root: Path, smoke: bool = False) -> list[Path]:
    raw_dir = root / "BCI-4-2A" / "raw_data"
    files = sorted(raw_dir.glob("A??[TE].mat"))
    files = [p for p in files if not any(part == "@eaDir" for part in p.parts)]
    if smoke:
        files = [p for p in files if p.name == "A01T.mat"]
    return files


def load_bcic2a_true_labels(root: Path, file_name: str) -> np.ndarray:
    label_path = root / "BCICIV2A_true_labels" / file_name
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    mat = loadmat(str(label_path), squeeze_me=True, struct_as_record=False)
    labels = np.asarray(mat["classlabel"]).astype(np.int64).reshape(-1)
    return labels


def scan_bcic2a_plan(root: Path, smoke: bool = False) -> DatasetPlan:
    if not root.exists():
        raise FileNotFoundError(root)
    files = bcic2a_a_files(root, smoke=smoke)
    rows: list[dict[str, Any]] = []
    label_mismatch_count = 0
    run_count = 0
    artifact_count = 0

    for mat_path in files:
        mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        runs = np.atleast_1d(mat["data"])
        true_labels = load_bcic2a_true_labels(root, mat_path.name)
        label_cursor = 0
        session_name = mat_path.stem
        subject_id = session_name[:3]
        session_type = session_name[-1]
        official_split = "train" if session_type == "T" else "eval"

        for run_index, run in enumerate(runs):
            trial = np.asarray(getattr(run, "trial", []), dtype=np.int64).reshape(-1)
            raw_y = np.asarray(getattr(run, "y", []), dtype=np.int64).reshape(-1)
            artifacts = np.asarray(getattr(run, "artifacts", []), dtype=np.int64).reshape(-1)
            if trial.size == 0:
                continue
            run_count += 1
            if artifacts.size == 0:
                artifacts = np.zeros_like(trial)
            if raw_y.size and true_labels[label_cursor : label_cursor + trial.size].size == trial.size:
                if not np.array_equal(raw_y, true_labels[label_cursor : label_cursor + trial.size]):
                    label_mismatch_count += int(
                        np.sum(raw_y != true_labels[label_cursor : label_cursor + trial.size])
                    )
            for trial_index_in_run, cue_sample_raw in enumerate(trial):
                true_label_raw = int(true_labels[label_cursor])
                label_id = true_label_raw - 1
                source_artifact = int(artifacts[trial_index_in_run]) if trial_index_in_run < artifacts.size else 0
                if source_artifact:
                    artifact_count += 1
                cue_sec = float(cue_sample_raw) / 250.0
                segment_start_sec = cue_sec + 2.0
                segment_end_sec = cue_sec + 6.0
                row = {
                    "sample_id": (
                        f"BCIC2A:{subject_id}:{session_name}:run{run_index:02d}:"
                        f"trial{trial_index_in_run:03d}"
                    ),
                    "dataset_key": "BCIC2A",
                    "subject_id": subject_id,
                    "session_id": session_name,
                    "trial_id": f"run{run_index:02d}_trial{trial_index_in_run:03d}",
                    "segment_id": f"{session_name}_run{run_index:02d}_trial{trial_index_in_run:03d}",
                    "source_relpath": relpath(mat_path, root),
                    "source_format": "mat",
                    "raw_inner_path": f"data[{run_index}].X",
                    "segment_start_sec": segment_start_sec,
                    "segment_end_sec": segment_end_sec,
                    "event_start_sec": cue_sec,
                    "event_end_sec": cue_sec,
                    "event_id": f"cue_{true_label_raw}",
                    "label_id": label_id,
                    "label_raw": str(true_label_raw),
                    "label_source": "BCICIV2A_true_labels/classlabel",
                    "channel_names_json": json.dumps(BCIC2A_CHANNELS),
                    "montage_name": "BCICIV2a_22_EEG_source_order",
                    "raw_sfreq": 250.0,
                    "notch_hz": 50.0,
                    "bandpass_high_effective": 75.0,
                    "split": "unassigned",
                    "segment_kind": "cue_2s_to_6s",
                    "qc_flags": QC_SOURCE_ARTIFACT if source_artifact else 0,
                    "run_id": f"run{run_index:02d}",
                    "trial_index_in_run": int(trial_index_in_run),
                    "trial_index_in_file": int(label_cursor),
                    "cue_sample_raw": int(cue_sample_raw),
                    "session_type": session_type,
                    "official_split": official_split,
                    "source_artifact": source_artifact,
                    "source_file_name": mat_path.name,
                    "channel_count": 22,
                }
                rows.append(row)
                label_cursor += 1
        if label_cursor != true_labels.size:
            raise ValueError(
                f"{mat_path.name}: raw trials={label_cursor}, true labels={true_labels.size}"
            )

    rows.sort(key=lambda r: (r["subject_id"], r["session_id"], r["trial_index_in_file"]))
    label_vocab = {
        str(k): {"name": v, "raw_aliases": [str(k + 1)]}
        for k, v in BCIC2A_LABEL_NAMES.items()
    }
    dataset_config = {
        "dataset_key": "BCIC2A",
        "source_root": "<provided-at-runtime>",
        "source_format": "mat",
        "file_scope": "A01-A09 T/E only; B* and S* excluded",
        "segment_seconds": 4.0,
        "target_sampling_rate": TARGET_SFREQ,
        "signal_unit": "uV",
        "notch_hz": 50.0,
        "bandpass_low": 0.3,
        "bandpass_high": 75.0,
        "channel_rule": "use first 22 columns of X as EEG; last 3 EOG channels excluded",
        "channel_names": BCIC2A_CHANNELS,
        "label_rule": "use true_labels/classlabel 1..4 and convert to 0..3",
        "segment_rule": "after preprocessing continuous run, take cue+[2s,6s)",
        "split_rule": "split=unassigned; official T/E session preserved as metadata",
        "label_mismatch_count_vs_raw_y": label_mismatch_count,
        "artifact_trial_count": artifact_count,
        "run_count": run_count,
        "qc_flags": qc_flag_mapping(),
    }
    return DatasetPlan(
        dataset_key="BCIC2A",
        rows=rows,
        c_max=22,
        t=800,
        segment_seconds=4.0,
        task_type="multiclass_classification",
        label_vocab=label_vocab,
        dataset_config=dataset_config,
        channel_count_distribution={"22": len(files)},
        expected_n_samples=288 * len(files),
    )


def write_bcic2a_zarr(root_path: Path, dataset_dir: Path, plan: DatasetPlan) -> dict[str, Any]:
    root = create_zarr_store(dataset_dir, plan)
    groups = row_groups_by_source(plan.rows)
    qc_counter: Counter[int] = Counter()
    label_counter: Counter[int] = Counter()
    start_time = time.time()

    for file_idx, (source_rel, indexed_rows) in enumerate(groups.items(), start=1):
        mat_path = root_path / source_rel
        logging.info(
            "BCIC2A %d/%d processing %s with %d samples",
            file_idx,
            len(groups),
            source_rel,
            len(indexed_rows),
        )
        mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        runs = np.atleast_1d(mat["data"])
        rows_by_run: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for idx, row in indexed_rows:
            rows_by_run[str(row["run_id"])].append((idx, row))

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

        for run_id, run_rows in sorted(rows_by_run.items()):
            run_index = int(run_id.replace("run", ""))
            run = runs[run_index]
            data_uv = np.asarray(run.X[:, :22], dtype=np.float64).T
            raw_sfreq = float(run.fs)
            processed, effective_high, file_qc = preprocess_continuous_uv(
                data_uv, raw_sfreq=raw_sfreq, notch_hz=50.0
            )
            for idx, row in run_rows:
                segment, valid_samples, segment_qc = extract_segment(
                    processed,
                    start_sec=float(row["segment_start_sec"]),
                    t_samples=plan.t,
                )
                signal, channel_mask, bad_mask, qc_flags = finalize_segment(
                    segment=segment,
                    c_max=plan.c_max,
                    t_samples=plan.t,
                    channel_names=BCIC2A_CHANNELS,
                    inherited_qc=int(row["qc_flags"]) | file_qc | segment_qc,
                    valid_time_samples=valid_samples,
                    min_expected_channels=22,
                )
                row["qc_flags"] = int(qc_flags)
                row["valid_time_samples"] = int(valid_samples)
                row["bandpass_high_effective"] = float(effective_high)
                indices.append(idx)
                signals.append(signal)
                channel_masks.append(channel_mask)
                bad_masks.append(bad_mask)
                channel_counts_list.append(22)
                labels.append(int(row["label_id"]))
                label_values.append(np.nan)
                subject_ids.append(str(row["subject_id"]))
                qc_flags_list.append(int(qc_flags))
                valid_time_list.append(int(valid_samples))
                qc_counter[int(qc_flags)] += 1
                label_counter[int(row["label_id"])] += 1
            del processed

        write_batch(
            root, indices, signals, channel_masks, bad_masks,
            channel_counts_list, labels, label_values, subject_ids,
            qc_flags_list, valid_time_list,
        )

    index_df = pd.DataFrame(plan.rows)
    write_parquet_compat(index_df, dataset_dir / "sample_index.parquet")
    write_json(dataset_dir / "label_vocab.json", plan.label_vocab)
    write_json(dataset_dir / "dataset_config.json", plan.dataset_config)
    summary = build_summary(plan, label_counter, qc_counter, elapsed_sec=time.time() - start_time)
    write_json(dataset_dir / "conversion_summary.json", summary)
    return summary
