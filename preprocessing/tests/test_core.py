from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import scipy.io
import zarr

from preprocessor.datasets.faced_new import C_MAX as FACED_CHANNELS
from preprocessor.datasets.seed import scan_seed_plan
from preprocessor.datasets.shu_mi import scan_shu_mi_plan
from preprocessor.dsp import preprocess_continuous_uv
from preprocessor.io_utils import relpath
from preprocessor.plan import DatasetPlan
from preprocessor.splits import apply_subject_disjoint_split
from preprocessor.validation import sample_order_sha256, validate_store
from preprocessor.writer import create_zarr_store, write_batch

ZARR_V2 = int(str(zarr.__version__).split(".", 1)[0]) < 3


class CorePreprocessorTests(unittest.TestCase):
    def test_paper_channel_counts(self) -> None:
        self.assertEqual(FACED_CHANNELS, 32)

    def test_seed_uses_trial_labels_and_ten_second_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scipy.io.savemat(root / "label.mat", {"label": np.asarray([[1, 0, -1]])})
            scipy.io.savemat(
                root / "1_20131027.mat",
                {
                    "subject_eeg1": np.zeros((62, 4000)),
                    "subject_eeg2": np.zeros((62, 4000)),
                    "subject_eeg3": np.zeros((62, 4000)),
                },
            )
            plan = scan_seed_plan(root)
            self.assertEqual((plan.c_max, plan.t), (62, 2000))
            self.assertEqual(
                [row["label_id"] for row in plan.rows],
                [2, 2, 1, 1, 0, 0],
            )
            self.assertTrue(
                all(row["segment_end_sec"] - row["segment_start_sec"] == 10.0
                    for row in plan.rows)
            )

    def test_shu_retains_all_32_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat_dir = Path(tmp) / "mat"
            mat_dir.mkdir()
            scipy.io.savemat(
                mat_dir / "sub-001_ses-01_task_motorimagery_eeg.mat",
                {
                    "data": np.arange(2 * 32 * 1000, dtype=np.float64).reshape(2, 32, 1000),
                    "labels": np.asarray([[1], [2]]),
                },
            )
            plan = scan_shu_mi_plan(Path(tmp))
            self.assertEqual(plan.c_max, 32)
            self.assertTrue(all(row["channel_count"] == 32 for row in plan.rows))

    def test_split_assignment_does_not_change_sample_order(self) -> None:
        rows = [
            {"sample_id": "sample-b", "subject_id": "b"},
            {"sample_id": "sample-a", "subject_id": "a"},
        ]
        before = [row["sample_id"] for row in rows]
        before_hash = sample_order_sha256(rows)
        apply_subject_disjoint_split(rows)
        self.assertEqual([row["sample_id"] for row in rows], before)
        self.assertEqual(sample_order_sha256(rows), before_hash)

    def test_physionet_bandpass_keeps_75_hz(self) -> None:
        rng = np.random.default_rng(0)
        signal = rng.normal(size=(2, 3200))
        processed, effective_high, flags = preprocess_continuous_uv(
            signal,
            raw_sfreq=160.0,
            notch_hz=60.0,
            bandpass_low=0.3,
            bandpass_high=75.0,
            target_sfreq=200.0,
        )
        self.assertEqual(processed.shape, (2, 4000))
        self.assertEqual(effective_high, 75.0)
        self.assertEqual(flags, 0)

    def test_relpath_rejects_source_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            root.mkdir()
            with self.assertRaises(ValueError):
                relpath(Path(tmp) / "outside.edf", root)

    @unittest.skipUnless(ZARR_V2, "Roundtrip test requires the pinned zarr<3")
    def test_zarr_roundtrip_and_validation(self) -> None:
        rows = [
            {"subject_id": "s1", "source_relpath": "a.edf"},
            {"subject_id": "s2", "source_relpath": "b.edf"},
        ]
        plan = DatasetPlan(
            dataset_key="Synthetic",
            rows=rows,
            c_max=2,
            t=20,
            segment_seconds=0.1,
            task_type="classification",
            label_vocab={"0": {"name": "a"}, "1": {"name": "b"}},
            dataset_config={"notch_hz": 50.0},
            channel_count_distribution={"2": 2},
            expected_n_samples=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            root = create_zarr_store(dataset_dir, plan)
            write_batch(
                root=root,
                indices=[0, 1],
                signals=[
                    np.zeros((2, 20), dtype=np.float32),
                    np.ones((2, 20), dtype=np.float32),
                ],
                channel_masks=[
                    np.ones(2, dtype=bool),
                    np.ones(2, dtype=bool),
                ],
                bad_masks=[
                    np.zeros(2, dtype=bool),
                    np.zeros(2, dtype=bool),
                ],
                channel_counts=[2, 2],
                labels=[0, 1],
                label_values=[np.nan, np.nan],
                subject_ids=["s1", "s2"],
                qc_flags=[0, 0],
                valid_time_samples=[20, 20],
            )
            with mock.patch(
                "preprocessor.validation.pd.read_parquet",
                return_value=pd.DataFrame(rows),
            ):
                checks = validate_store(dataset_dir, plan)
            self.assertEqual(checks["signals_shape"], [2, 2, 20])

    def test_empty_zarr_batch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            write_batch(
                root=None,
                indices=[],
                signals=[],
                channel_masks=[],
                bad_masks=[],
                channel_counts=[],
                labels=[],
                label_values=[],
                subject_ids=[],
                qc_flags=[],
                valid_time_samples=[],
            )


if __name__ == "__main__":
    unittest.main()
