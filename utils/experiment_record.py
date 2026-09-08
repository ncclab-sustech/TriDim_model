# utils/experiment_record.py

import os
import time
import socket
import subprocess
from pathlib import Path


def _run_cmd(cmd):
    """Run a shell command and return stripped stdout. Return 'unknown' on failure."""
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            shell=True,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _yaml_value(value):
    """Format a Python value as a simple YAML scalar."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.6f}"
    text = str(value)
    # Quote strings that may contain special YAML characters.
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "\"", "'"]) or text.strip() != text:
        text = text.replace('"', '\\"')
        return f'"{text}"'
    return text


def write_simple_yaml(path, record):
    """Write a flat dictionary as a simple YAML file without requiring PyYAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for key, value in record.items():
            f.write(f"{key}: {_yaml_value(value)}\n")

def get_job_id():
    """Get job id from common schedulers. Fall back to timestamped local id."""
    for key in [
        "LSB_JOBID",       # LSF / bsub
        "SLURM_JOB_ID",    # Slurm
        "PBS_JOBID",       # PBS / Torque
        "JOB_ID",          # SGE
    ]:
        value = os.environ.get(key)
        if value:
            return str(value)

    return "local_" + time.strftime("%Y%m%d_%H%M%S")

def collect_experiment_record(
    args,
    setting,
    test_metrics,
    val_metrics=None,
    result_dir="results/runs",
    bsub_script=None,
    notes="",
):
    """
    Collect metadata and metrics for one experiment run.

    Args:
        args: argparse Namespace.
        setting: experiment setting string used by the training code.
        test_metrics: dict returned by exp.test().
        val_metrics: optional validation metrics dict.
        result_dir: directory to save YAML files.
        bsub_script: optional path to the bsub script.
        notes: optional experiment notes.

    Returns:
        record: dict
        out_path: Path
    """
    job_id = get_job_id()
    branch = _run_cmd("git branch --show-current")
    commit = _run_cmd("git rev-parse --short HEAD")
    git_status = _run_cmd("git status --short")

    dataset = getattr(args, "data", "unknown")
    model = getattr(args, "model", "unknown")
    seed = getattr(args, "seed", getattr(args, "seed_start", "unknown"))
    itr = getattr(args, "itr", "unknown")

    queue = os.environ.get("LSB_QUEUE", "local")
    node = socket.gethostname()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    dataset_yaml = getattr(args, "dataset_paths_yaml", "")
    gpu_request = os.environ.get("LSB_GPU_REQ", "")

    train_log = f"logs/{job_id}.train.log" if job_id != "local" else ""

    val_metrics = val_metrics or {}

    record = {
        "job_id": job_id,
        "owner": os.environ.get("USER", "unknown"),
        "date": time.strftime("%Y-%m-%d"),
        "time": time.strftime("%H:%M:%S"),
        "branch": branch,
        "commit": commit,
        "git_dirty": bool(git_status),
        "model": model,
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "seed_start": getattr(args, "seed_start", ""),
        "itr": itr,
        "queue": queue,
        "node": node,
        "cuda_visible_devices": cuda_visible,
        "gpu_request": gpu_request,
        "config": dataset_yaml,
        "bsub_script": bsub_script or "",
        "train_log": train_log,
        "val_accuracy": val_metrics.get("Accuracy", None),
        "val_precision": val_metrics.get("Precision", None),
        "val_recall": val_metrics.get("Recall", None),
        "val_f1": val_metrics.get("F1", None),
        "val_auroc": val_metrics.get("AUROC", None),
        "val_auprc": val_metrics.get("AUPRC", None),
        "test_accuracy": test_metrics.get("Accuracy", None),
        "test_precision": test_metrics.get("Precision", None),
        "test_recall": test_metrics.get("Recall", None),
        "test_f1": test_metrics.get("F1", None),
        "test_auroc": test_metrics.get("AUROC", None),
        "test_auprc": test_metrics.get("AUPRC", None),
        "notes": notes,
    }

    safe_model = str(model).replace("/", "_")
    safe_dataset = str(dataset).replace("/", "_")
    safe_seed = str(seed).replace("/", "_")
    filename = f"{job_id}_{safe_dataset}_{safe_model}_seed{safe_seed}.yaml"

    out_path = Path(result_dir) / filename
    write_simple_yaml(out_path, record)

    return record, out_path