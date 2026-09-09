# TriDim — paper Full benchmark (8:1:1, 57.70%)

Portable reproduction snapshot for the Full model in **Beyond Flattened Tokens:
Structure-Preserving EEG Decoding with Reusable TriDim Blocks**.

This snapshot targets the original eight-dataset, three-seed paper table:
**Full macro accuracy = 57.70%**. It does not substitute the later five-seed,
six-dataset or LOSO experiments.

## Protocol

- Model: `tridim` (Full).
- Historical records retain their original model label; `reported/protocol.json`
  maps that label to the current module name. The model implementation and weights
  are unchanged by the filename and import rename.
- Split/initialization seeds: **5, 42, 43**, three runs per dataset, 24 total.
- Nominal train/validation/test ratio: **8:1:1**, rounded at the splitting-unit level.
- Checkpoint selection: **validation Accuracy**; final held-out test evaluation.
- Eight dataset means are equally weighted; samples are not pooled across datasets.

## Reported Full results

Mean +/- sample standard deviation across seeds 5/42/43, in percent:

| Dataset | Accuracy |
|---|---:|
| AD65 | 63.83 +/- 6.10 |
| SleepEDF | 85.61 +/- 2.19 |
| BCI-IV-2A | 56.60 +/- 2.71 |
| SHU-MI | 60.20 +/- 7.16 |
| PhysioNet-MI | 60.43 +/- 8.00 |
| FACED | 53.04 +/- 2.87 |
| SEED | 52.87 +/- 2.22 |
| SEED-V | 29.04 +/- 4.34 |
| **Eight-dataset macro average** | **57.70** |

The unrounded macro mean is **57.7009041667%**. Recompute it without EEG data
or PyTorch:

```bash
python scripts/verify_reported_results.py
```

`reported/full_24_runs.json` contains the 24 archived metrics and SHA256
references to the source records and archived logs. The 24 log copies are
in `logs/full/`; these are the retained sanitized per-seed sections, not newly
generated training output. Every final log accuracy
was checked against its record during assembly (tolerance 1e-5).

## Install and prepare data

Use Python 3.10/3.11 and a compatible PyTorch environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TRIDIM_DATA_ROOT=/path/to/preprocessed/eeg
python scripts/smoke_test_models.py
```

See [data preparation](data/README.md), [dataset directory names](data/datasets.csv)
and [raw-data converters](preprocessing/README.md). Raw EEG and weights are
not included. Obtain each dataset through its original distributor.
SEED-V and SleepEDF use the six provided fixed manifests. Their row indices
require matching preprocessing and sample order; changing sample order requires
regenerating and validating manifests, not blindly reusing these files.

## Reproduce the paper Full grid

Run commands from the repository root. First inspect all 24 jobs without training:

```bash
python scripts/run_full_paper.py --gpus 0 1 2 3 --dry-run
```

Then run the complete grid on four visible GPUs:

```bash
python scripts/run_full_paper.py --gpus 0 1 2 3
```

Or one dataset on one GPU (all three paper seeds):

```bash
python scripts/run_full_paper.py --datasets AD65 --gpus 0
```

Each dataset/seed has an isolated working directory under `runs/<timestamp>/`,
including its effective config, command, training log, checkpoint and results.
Four GPUs execute four independent single-GPU jobs. Full training parameters
come unchanged from `configs/paper/full/*.yaml`; the launcher only resolves paths,
selects one original seed per process and assigns GPU/work directories.

## Scope and provenance limits

The model, training and loader files were taken from the retained historical
portable release; the training settings are retained, with the model identifier standardized to `tridim`. The new launcher
adds run isolation. Manifest source-path metadata is made portable without
changing sample indices. The model registry and smoke check contain only Full;
this repository publishes only the Full paper result grid.

Historical result records mark the training Git commit as `unknown`. Therefore
this release hash identifies the assembled snapshot and does not prove an
original training commit. Archived metrics can be recomputed; fresh training
is not guaranteed to produce identical numbers across hardware/software.
The 24 long training jobs were not rerun as part of packaging this release.

This snapshot is not a new license grant and does not resolve licensing for
the entire legacy repository. Repository visibility remains private unless
the owners separately decide to publish it after their release review.
