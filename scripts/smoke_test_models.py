#!/usr/bin/env python3
"""Run synthetic forward/backward checks for the released Full model."""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "paper" / "full" / "SEED_full.yaml"
EXPECTED_SEED_FULL_PARAMS = 3_251_600
MODEL_MODULES = ["tridim"]


def load_model_config(path: Path, seq_len: int, num_class: int) -> SimpleNamespace:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = dict(payload["params"])
    params.update(
        seq_len=seq_len,
        enc_in=int(payload["electrode_channel_count"]),
        num_class=num_class,
        output_mode="classification",
    )
    return SimpleNamespace(**params)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def smoke_test(
    module_name: str,
    config: SimpleNamespace,
    batch_size: int,
) -> tuple[int, int]:
    module = importlib.import_module(f"models.{module_name}")
    model = module.Model(config)
    model.train()

    inputs = torch.randn(batch_size, config.seq_len, config.enc_in)
    targets = torch.arange(batch_size, dtype=torch.long) % config.num_class
    logits = model(inputs)

    expected_shape = (batch_size, config.num_class)
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected_shape:
        actual = type(logits).__name__
        if isinstance(logits, torch.Tensor):
            actual = str(tuple(logits.shape))
        raise AssertionError(
            f"{module_name}: output={actual}, expected tensor {expected_shape}"
        )
    if not torch.isfinite(logits).all():
        raise AssertionError(f"{module_name}: non-finite forward output")

    loss = F.cross_entropy(logits, targets)
    loss.backward()
    finite_gradients = [
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not finite_gradients or not all(finite_gradients):
        raise AssertionError(f"{module_name}: missing or non-finite gradients")

    return count_parameters(model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=240,
        help="Synthetic sequence length; 240 yields three SEED patches.",
    )
    parser.add_argument("--num-class", type=int, default=3)
    parser.add_argument(
        "--expected-full-params",
        type=int,
        default=EXPECTED_SEED_FULL_PARAMS,
        help="Set to 0 to disable the paper Full-model parameter-count check.",
    )
    args = parser.parse_args()

    if args.batch_size < 1 or args.seq_len < 1 or args.num_class < 2:
        parser.error("batch size and sequence length must be positive; num-class >= 2")

    config_path = args.config.resolve()
    config = load_model_config(config_path, args.seq_len, args.num_class)
    torch.manual_seed(0)
    torch.set_num_threads(1)

    print(f"Config: {config_path.relative_to(ROOT)}")
    print(
        f"Synthetic input: batch={args.batch_size}, time={args.seq_len}, "
        f"channels={config.enc_in}, classes={args.num_class}"
    )

    full_total = None
    for module_name in MODEL_MODULES:
        total, trainable = smoke_test(module_name, config, args.batch_size)
        if module_name == MODEL_MODULES[0]:
            full_total = total
        print(
            f"PASS {module_name}: total={total:,}, "
            f"trainable={trainable:,}"
        )

    if args.expected_full_params and full_total != args.expected_full_params:
        raise AssertionError(
            f"Full parameter count is {full_total:,}; "
            f"expected {args.expected_full_params:,}"
        )
    if full_total is None or not math.isfinite(float(full_total)):
        raise AssertionError("Full parameter count was not computed")

    print(
        f"All {len(MODEL_MODULES)} model passed forward/backward checks. "
        f"Full={full_total:,} parameters."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
