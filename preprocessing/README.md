# EEG preprocessing

The public converter covers the eight downstream datasets used in the paper:
AD65, BCI-IV-2A, FACED, PhysioNet-MI, SEED, SEED-V, SHU-MI, and SleepEDF.

The converter resamples EEG to 200 Hz, applies a 0.3 Hz high-pass and a
source-supported low-pass capped at 75 Hz, applies the source-specific
power-line notch, and stores microvolt-valued `float32` segments without
dataset-level normalization. `InstanceTimeNorm` performs
per-sample, per-channel temporal normalization inside the model.

## Setup

```bash
cd preprocessing
pip install -r requirements.txt
```

## Build

```bash
python run.py --datasets ad65,bcic2a,faced_new,physionet_mi,seed,seed_v_10s,shu_mi,sleepedf \
  --out-root /path/to/TRIDIM_DATA_ROOT \
  --ad65-root /path/to/AD65 \
  --bcic2a-root /path/to/BCIC2A \
  --faced_new-root /path/to/FACED \
  --physionet_mi-root /path/to/eegmmidb \
  --seed-root /path/to/SEED \
  --seed_v_10s-root /path/to/SEED-V \
  --shu_mi-root /path/to/SHU-MI \
  --sleepedf-root /path/to/SleepEDF
```

Every selected dataset requires an explicit raw root. Use `--dry-run` to scan
and summarize without writing, `--smoke` for a deterministic small subset, and
`--overwrite` only when an existing processed dataset should be replaced.

## Output schema

Each processed directory contains:

- `<dataset>.zarr`, with signals shaped `[sample, channel, time]`;
- `sample_index.parquet`, including labels and subject/session provenance;
- `dataset_config.json` and `label_vocab.json`;
- conversion and validation summaries;
- `logs/conversion.log`.

`conversion_summary.json` and the Zarr attributes include a
`sample_order_sha256` fingerprint. Index-based split manifests are valid only
when this fingerprint matches the store from which they were generated.

The training loader converts Zarr signals to `[sample, time, channel]` before
passing them to TriDimEEG.

## Validation

```bash
cd preprocessing
python -m unittest discover -s tests -v
```

These tests cover DSP limits, path containment, paper input shapes, SEED label
mapping, source-order stability, Zarr writing, and schema validation using
synthetic data. Dataset licenses generally prevent redistributing raw EEG, so
users must acquire each source release separately.
