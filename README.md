# TriDim — paper Full version: 8:1:1, 57.70%

The paper benchmark uses **Full TriDim**, **eight datasets**, and **seeds
5/42/43**, with nominal **8:1:1** train/validation/test splits and
**validation Accuracy** checkpoint selection.

**Use the isolated paper snapshot below. The old root-level `run.py`,
`configs/` and historical model files are retained development material;
they are not the entry point for reproducing this paper table.**

## Download the runnable paper snapshot

- [TriDim_paper_811_full_5770.zip](TriDim_paper_811_full_5770.zip)
- [SHA256 checksum](paper_5770_sha256.txt)

```bash
unzip TriDim_paper_811_full_5770.zip
cd TriDim_paper_811_full_5770
python scripts/verify_reported_results.py
```

The archive contains model/training/loader code, all eight Full configs,
the six original-seed SleepEDF/SEED-V manifests, preprocessing code,
24 structured paper-result records, a table verification script, model
smoke tests, and an isolated single-/four-GPU Full launcher.

## Paper Full accuracy

Mean +/- sample standard deviation over seeds 5/42/43 (%):

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

The mean of the eight unrounded dataset means is **57.7009041667%**.
These are the original three-seed paper results, not the later five-seed
extension or LOSO results.

## Train from the extracted snapshot

Use Python 3.10/3.11 and install the bundled requirements. Follow the archive's
`data/README.md` and `preprocessing/README.md` to prepare the eight datasets.

```bash
pip install -r requirements.txt
export TRIDIM_DATA_ROOT=/path/to/preprocessed/eeg
python scripts/smoke_test_models.py
python scripts/run_full_paper.py --gpus 0 1 2 3 --dry-run
python scripts/run_full_paper.py --gpus 0 1 2 3
```

These commands must run inside `TriDim_paper_811_full_5770/` after extraction.
Each dataset/seed runs in its own working directory, with separate config,
checkpoint and log. No EEG data or model weights are bundled.

## Protocol and provenance qualifications

- 8:1:1 is nominal, rounded at the split-unit level. SleepEDF retains the
  paper's **122/15/16-session** split and train-only t40 cap. Participants can
  occur in multiple subsets through different nights; this is not a strict
  participant-independent evaluation.
- The 24 archived result values were checked against final log accuracies.
  Original record/log hashes are included. Historical training commit fields
  are `unknown`; the assembled snapshot is not claimed to be a verified
  original training commit.
- Fresh training may differ across hardware and library versions. 57.70% is
  the archived paper mean, not a promised result of every rerun.
- This update packages the Full paper version. It does not finish cleaning
  the legacy root tree or its Git history, and does not change repository
  visibility or grant a new license.
