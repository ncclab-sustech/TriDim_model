#!/usr/bin/env python3
"""CLI entry point for building canonical downstream Zarr stores.

Usage examples:
    python run.py --datasets ad65 --ad65-root /raw/ad65 --out-root /processed
    python run.py --datasets bcic2a --bcic2a-root /raw/bcic2a --dry-run
    python run.py --datasets physionet_mi --physionet_mi-root /raw/eegmmidb --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from preprocessor.datasets import DATASET_REGISTRY
from preprocessor.io_utils import clean_existing_output, write_json
from preprocessor.logging_utils import (
    attach_file_logger,
    configure_console_logging,
    detach_file_logger,
)
from preprocessor.validation import validate_store


def build_parser() -> argparse.ArgumentParser:
    valid_keys = ", ".join(sorted(DATASET_REGISTRY))
    parser = argparse.ArgumentParser(
        description="Build canonical downstream Zarr stores.",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help=f"Comma-separated dataset list: {valid_keys}",
    )
    for key, spec in sorted(DATASET_REGISTRY.items()):
        parser.add_argument(
            f"--{key}-root",
            default=spec["default_root"],
            help=spec["help"],
        )
    parser.add_argument(
        "--out-root",
        default="",
        help="Output root directory (required unless set via --out-root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan inputs and print plans without writing Zarr stores.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Process a small deterministic subset for smoke validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing dataset output directory before writing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser


def get_root(args: argparse.Namespace, key: str) -> Path:
    attr = f"{key}_root"
    value = str(getattr(args, attr) or "").strip()
    if not value:
        raise ValueError(f"--{key}-root is required when processing '{key}'.")
    return Path(value).expanduser()


def run_dry_run(datasets: list[str], args: argparse.Namespace) -> None:
    summary = {}
    for key in datasets:
        spec = DATASET_REGISTRY[key]
        root = get_root(args, key)
        plan = spec["scan"](root, smoke=args.smoke)
        label_counts = Counter(int(row["label_id"]) for row in plan.rows)
        source_count = len({str(row["source_relpath"]) for row in plan.rows})
        summary[key] = {
            "dataset_key": plan.dataset_key,
            "n_samples": len(plan.rows),
            "expected_n_samples": plan.expected_n_samples,
            "n_source_files": source_count,
            "n_subjects": len({str(row["subject_id"]) for row in plan.rows}),
            "c_max": plan.c_max,
            "t": plan.t,
            "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
            "channel_count_distribution": plan.channel_count_distribution,
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def run_dataset(key: str, args: argparse.Namespace) -> None:
    spec = DATASET_REGISTRY[key]
    root = get_root(args, key)
    out_root = Path(args.out_root)
    plan = spec["scan"](root, smoke=args.smoke)
    dataset_dir = out_root / spec["output_dir_name"]
    clean_existing_output(dataset_dir, overwrite=args.overwrite)
    handler = attach_file_logger(dataset_dir / "logs" / "conversion.log")
    try:
        logging.info("Writing %s to %s", plan.dataset_key, dataset_dir)
        summary = spec["write"](root, dataset_dir, plan)
        validation = validate_store(dataset_dir, plan)
        write_json(dataset_dir / "validation_summary.json", validation)
        logging.info("%s summary: %s", plan.dataset_key, summary)
        logging.info("%s validation: %s", plan.dataset_key, validation)
    finally:
        detach_file_logger(handler)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_console_logging(args.log_level)
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    invalid = sorted(set(datasets) - set(DATASET_REGISTRY))
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}. Valid: {sorted(DATASET_REGISTRY)}")

    if args.dry_run:
        run_dry_run(datasets, args)
        return 0

    if not str(args.out_root or "").strip():
        raise SystemExit("--out-root is required (no default path is baked in).")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    for key in datasets:
        run_dataset(key, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
