# eeg_basis_mixer_v2_nobasis (release)

Single-model release of the **nobasis** TriAxis mixer baseline used in our
EEG cross-subject classification experiments. This package is intentionally
minimal: only one model (`eeg_basis_mixer_v2_nobasis`) and one data loader,
covering 19 public EEG datasets through a unified subject-wise h5 layout.

## 1. Layout

```
TeCh_nobasis_release/
├── run.py                          # entry point (argparse + yaml override)
├── requirements.txt
├── configs/
│   ├── datasets/<dataset>.yaml     # one file per dataset, sets root_path + hparams
│   └── electrodes/<dataset>.csv    # 3D electrode coordinates (used by channel adapter)
├── data_provider/
│   ├── data_factory.py             # routes every dataset to ADHDLoader
│   ├── data_loader.py              # subject-wise h5 loader, 3 split modes
│   └── uea.py                      # collate_fn / normalization
├── exp/
│   ├── exp_basic.py                # device + model registry
│   └── exp_classification.py       # train / val / test loop
├── layers/Augmentation.py          # jitter / scale / flip / mask augmentations
├── models/eeg_basis_mixer_v2_nobasis.py
├── utils/tools.py                  # cosine LR, EarlyStopping
└── scripts/run_faced_nobasis_xsub.sh
```

## 2. Install

```bash
conda create -n nobasis python=3.10 -y
conda activate nobasis
pip install -r requirements.txt
```

The pinned versions correspond to PyTorch 2.4.1 + CUDA 12. Adjust the torch
build to match your CUDA driver if needed.

## 3. Data format

All 19 supported datasets share one h5 schema. Each subject is one file under
`<root_path>/`. The loader auto-detects the naming convention; the following
patterns all work without code changes:

| Style                                  | Examples                                | Datasets                          |
|----------------------------------------|-----------------------------------------|-----------------------------------|
| `sub_<int>.h5`                         | `sub_001.h5`, `sub_42.h5`               | most datasets                     |
| `sub-<int>.h5`                         | `sub-1.h5`                              | older preprocessing               |
| `sub-<gender><int>.h5`                 | `sub-f1.h5`, `sub-m2.h5`                | FACED-style gendered IDs          |
| `sub-<int>_task-...eeg.h5`             | `sub-001_task-eyesclosed_eeg.h5`        | AD65                              |
| `S<int>.h5`                            | `S001.h5`                               | Physionet_MI                      |
| `sub<int>.h5` (no separator)           | `sub000.h5`                             | FACED_new                         |
| `sub_SC<int><tail>.h5`                 | `sub_SC4001E0.h5`                       | sleep-cassette-200hz              |
| `A<int>[TE].h5`                        | `A01T.h5`, `A01E.h5`                    | BCIC2A (T/E treated as separate)  |
| `<H\|MDD> S<int> <EC\|EO\|TASK>.h5`    | `MDD S1 EC.h5`, `H S15 EO.h5`           | MDD                               |

Inside each h5:

```
sub_001.h5
└── trial_<i>           (h5.Group)
    └── segment_<j>     (h5.Group, attrs may carry "label")
        ├── eeg         (h5.Dataset, shape (C, T) or (T, C); attrs may carry "label")
        └── label       (optional dataset, used if attrs absent)
```

Channel order in `eeg` must match the rows of
`configs/electrodes/<dataset>.csv` (columns: `name, x, y, z` in head-coordinate
metres). Update the CSV if you preprocess with a different montage.

## 4. Run

The 5-seed cross-subject (4:3:3) sweep is the default configuration. CLI args
override yaml values. Minimal invocation（不要使用augmentations）:

```bash
python run.py \
    --model eeg_basis_mixer_v2_nobasis \
    --data FACED_new \
    --dataset_paths_yaml ./configs/datasets/FACED_new.yaml \
    --root_path /your/path/to/FACED_new \
    --gpu 0 --gpu_idx 0 --num_workers 4 \
    --itr 5 --seed_start 42 \
    --split_mode label_order --train_ratio 0.4 --val_ratio 0.3 \
    --augmentations none --select_metric F1
```

There is also `scripts/run_faced_nobasis_xsub.sh` you can copy as a template
for other datasets — change `--data`, `--dataset_paths_yaml`, and `--root_path`.

For deterministic cuBLAS:

```bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

(`run.py` already sets this if not present in the environment.)

## 5. Split modes

这里严格使用跨被试的，然后数据划分方式是4:3:3

| `--split_mode`               | Behavior                                               |
|------------------------------|--------------------------------------------------------|
| `label_order` *(default)*    | Deterministic subject split; ordered by class then ID. Used in the paper. |
| `stratified_random`          | Per-class random subject split, seeded by `--seed_start + itr_idx`. |
| `segment_stratified_random`  | Splits **segments** instead of subjects (NOT cross-subject; for ablations only). |

Defaults: `train_ratio=0.4`, `val_ratio=0.3`, test gets the rest.

## 6. Output

After all `--itr` seeds finish, `run.py` prints mean/std over six metrics:
Accuracy, Precision, Recall, F1, AUROC, AUPRC. Per-seed checkpoints are
written under `./checkpoints/<setting>/` and removed after testing. Per-seed
metric jsons land under `./results/`.

## 7. Notes for porting

* `--downstream_root /shared/dir` lets you point the loader at a single
  parent directory containing `<dataset>/` subfolders, instead of setting
  `--root_path` per run.
* `--use_channel_adapter` enables the RBF geometric prior over electrodes.
  It assumes both the input montage CSV and the canonical montage CSV are
  set (yaml does this for you).
* `--v_layer` is accepted for yaml compatibility but unused by the nobasis
  model.
