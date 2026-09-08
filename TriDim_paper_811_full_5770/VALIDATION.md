# Packaging validation — 2026-09-08

- 24 unique archived dataset/seed records: verified against final log accuracies.
- Eight-dataset mean: 57.7009041667%, displayed as 57.70%.
- Eight original Full configurations: seeds5/42/43, ratios0.8/0.1, validation Accuracy; source bytes preserved.
- Six fixed manifests: sample indices unchanged; only source-path metadata made portable.
- Python syntax checks passed for bundled source files.
- Full launcher dry-run generated 24 distinct dataset/seed jobs and isolated directories.
- Linux CPU synthetic forward/backward checks passed for all 13 registry model variants; SEED Full parameter count3,251,600. Validation host used PyTorch2.10.0; this is a functional smoke check, not a reproduction of H20 benchmark scores.
- Final archive inventory and SHA256 verification passed before upload.
- No new full-dataset training was performed for this package. Installation from an empty environment and raw EEG conversion were not repeated during packaging.

Original training commit IDs remain unknown in the archived result records. No claim of exact original commit provenance is made.
