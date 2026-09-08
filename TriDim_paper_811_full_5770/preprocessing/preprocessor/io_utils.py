from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_parquet_compat(df: pd.DataFrame, path: Path) -> None:
    """Write parquet through an in-memory buffer for restrictive mounts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    path.write_bytes(buf.getvalue())


def clean_existing_output(dataset_dir: Path, overwrite: bool) -> None:
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dataset_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "logs").mkdir(parents=True, exist_ok=True)


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise ValueError(
            f"Source path must be inside the supplied dataset root: {path}"
        ) from exc
