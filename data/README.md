# Data preparation

Raw EEG is not included. Download each dataset from its official distributor
and follow its license and data-use terms. Converters for all eight paper
datasets are available under `preprocessing/`. See
[../preprocessing/README.md](../preprocessing/README.md).

Set the common parent directory:

```bash
export TRIDIM_DATA_ROOT=/path/to/preprocessed/eeg
```

The expected subdirectory names and processed tensor statistics are listed in
`datasets.csv`.

## AD65 labels

AD65 retains all three diagnostic groups:

| Label | Group |
|---:|---|
| 0 | Control |
| 1 | Frontotemporal dementia |
| 2 | Alzheimer's disease |

It is therefore a three-class task with a uniform-class chance level of
33.3%, not a binary task with a 50% chance level. The verified conversion
contains 88 subjects, 19 channels, and 6,938 non-overlapping 10-s samples at
200 Hz. Machine-readable counts and label definitions are under
`metadata/AD65/`.

## Submitted preprocessing protocol

- all recordings are resampled to 200 Hz;
- recordings are high-pass filtered at 0.3 Hz and low-pass filtered at up to
  75 Hz, subject to the source sampling rate and upstream preprocessing;
- a notch filter is applied at the source-specific power-line frequency;
- AD65, FACED, SEED, and SEED-V use 10-s windows;
- BCI-IV-2A, SHU-MI, and PhysioNet-MI use 4-s windows;
- SleepEDF uses 30-s epochs;
- per-sample, per-channel normalization is performed inside the model by
  `InstanceTimeNorm`; no global normalization statistics are required.
