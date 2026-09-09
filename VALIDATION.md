# Full-only root layout — 2026-09-08

The repository contains the paper Full model, eight original 8:1:1 configurations,
six fixed manifests, required training/preprocessing code, and the 24 archived
Full logs for seeds 5, 42 and 43. Legacy 4:3:3, pretraining placeholders, checkpoints,
backups and ablation implementations are omitted from the current tree.

- Full model implementation bytes and training settings are unchanged.
- The module filename, imports and configured model identifier are now `tridim`;
  historical logs and metric records retain their original labels.
- The runner, model imports and smoke-test registry now expose only Full.
- The archived log bytes match the pre-existing result-record SHA256 values.
- The verifier checks every log's final test accuracy and recomputes the eight-
  dataset macro mean: 57.7009041667% (57.70%).
- The root-layout change is gated on Python syntax, a 24-job launcher dry run,
  log/hash verification, Full CPU forward/backward and import checks.
- The earlier 13-model smoke check applied to the previous package; only Full
  is retained here. Expected SEED Full parameter count: 3,251,600.
- No new full-dataset training is performed by this repository cleanup.

The root README is preserved at the owner's request and still describes the
legacy version. Current execution instructions are in PAPER_PROTOCOL.md.
Historical training commit IDs remain unknown.
Archived logs are the existing sanitized per-seed copies; their provenance headers
are retained unchanged.
