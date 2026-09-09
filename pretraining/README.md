# TriDim pretraining integrations

This directory restores our CBraMod+TriDim, CSBrain+TriDim and REVE+TriDim
pretraining implementations as browsable source. These are separate experiments
from the supervised eight-dataset Full result (57.70%) at the repository root.
They do not replace the root model, configurations, seeds or Full logs.

## Included

- Pinned upstream revisions and the required local pretraining overlays.
- Explicit TUH2k presets and a portable launcher recording the effective command.
- The retained CSBrain four-GPU DDP entry.
- Archived convergence data with limitations, checkpoint reference hashes, and
  per-source provenance. No weight binaries or EEG recordings are uploaded.

The old CSBrain and CBraMod placeholder directories are not the source of this
package; these files come from retained experiment/release material.

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
The code caps the sorted H5 file list at `max_subjects=2000`. The historical
"TUH2k" label therefore does not mean exactly 2,000 training windows, and a file
stem is not independent proof of participant identity. Matching the original
data file set and ordering remains necessary; no verified TUH2k file-list
manifest is included in this package.

## Module naming

The shared TriDim module is `overlays/cbramod/tridim.py`. CBraMod and REVE
import it as `tridim`. This is a filename/import rename with unchanged model
implementation and state-dict keys. The source manifest retains original
source paths and hashes alongside the current publication hashes.

## Provenance limits

`overlays/reve/src/utils/__init__.py` is a publication portability fix that
prevents CBraMod's `utils` package from shadowing REVE's initialization helpers.
It does not change the pretraining model or hyperparameters.

`overlays/csbrain/pretrain_trainer_h5.py` is explicitly a portable reconstruction,
not a recovered original trainer. The retained DDP entry imports its mask helper,
which delegates to pinned CSBrain upstream code. `SOURCE_MANIFEST.json` identifies
the retained files and source hashes. A functional test does not establish an
exact historical training commit or reproduce full-dataset scores.

Seed 42 is explicit in the presets, derived from retained jobs/defaults; it is
not a claim that all archived results contain independently verified run seeds.
These pretraining settings are separate from the downstream seeds 5/42/43.

`evidence/SOURCES.md` describes reconstructed curve points and missing epochs.
`evidence/checkpoints.json` is reference metadata only; its files are not hosted
here. In particular, the archived REVE+TriDim weight corresponds to epoch 39,
whereas the preset runs for 40 epochs.
