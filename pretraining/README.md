# TriDim pretraining integrations

This directory restores our CBraMod+TriDim, CSBrain+TriDim and REVE+TriDim
pretraining implementations as browsable source. These are separate experiments
from the supervised eight-dataset Full result (57.70%) at the repository root.
They do not replace the root model, configurations, seeds or Full logs.

## Included

- Pinned upstream revisions and the required local pretraining overlays.
- Explicit TUH2k presets and a portable launcher recording the effective command.
- The retained CSBrain four-GPU DDP entry.

## Setup

Use a separate Python 3.11 environment and install an appropriate PyTorch build.
From the repository root:

```bash
pip install -r pretraining/requirements.txt
python pretraining/prepare_upstreams.py
python pretraining/verify.py
python pretraining/smoke_test.py --upstream-root pretraining/upstream
```

The preparation script clones the revisions in `upstream.json` and copies the
matching overlays. It refuses an existing destination to protect local work.
CSBrain is obtained directly from upstream; see `THIRD_PARTY_NOTICES.md`.

## Training

The launcher defaults to the GPU counts in the retained launch scripts:

| Integration | GPU processes | Batch per process | Epochs | Mask ratio |
|---|---:|---:|---:|---:|
| CBraMod+TriDim | 1 | 128 | 40 | 0.50 |
| CSBrain+TriDim | 4 | 64 | 40 | 0.50 |
| REVE+TriDim | 4 | 16 | 40 | 0.55 |

```bash
python pretraining/run.py --family cbramod --data-root /path/to/TUH2k_h5 --dry-run
python pretraining/run.py --family csbrain --data-root /path/to/TUH2k_h5 --dry-run
python pretraining/run.py --family reve --data-root /path/to/TUH2k_h5 --dry-run
```

Remove `--dry-run` to train. Each run writes `launch.json`, `train.log`, checkpoints
and `exit_code.txt` beneath `runs/pretraining/`. `--gpus 0 1 2 3` overrides the
device list; changing its length changes global batch size and is not the same
training setting as the historical preset. Existing output directories are refused.

Input is the retained `sub_*.h5` format, with EEG under
`trial_*/segment_*/eeg`, 21 channels, 200 Hz, and 6,000-point windows.



