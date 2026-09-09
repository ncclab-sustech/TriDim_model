# Convergence evidence

`pretrain_convergence_data.csv` is copied byte-for-byte from the retained
2026-08-31 audit. It contains official-versus-TriDim series for three host models.
These are archived extracted data, not full training logs generated for this release.

- REVE: per-epoch Training Loss entries from run summary logs; 40 epochs for
  each variant. Duplicate resumed epoch entries (official 24, TriDim 23) were
  resolved using the later value matching the saved checkpoint name.
- CBraMod and CSBrain: reconstructed from checkpoint filenames. They must not
  be described as complete recovered epoch logs.
- Both CBraMod series lack epoch 39. The official epoch-1 duplicate was resolved
  using the later checkpoint modification time. Missing values are not imputed.
- Both CSBrain series start at epoch 2. The prior audit treated epoch-1 values
  15608.307802 and 4447.240898 as unnormalized reporting artifacts and excluded
  them. This is a documented historical audit interpretation.
- The prior audit filtered an abandoned CSBrain checkpoint-size series and a
  separate CBraMod continuation series from mixed directories.
- Archived REVE+TriDim selects epoch 39 (0.096649), not epoch 40 (0.096660).
  REVE official ends at 0.138687; its per-process batch was 32 versus 16 for
  TriDim. Do not assume identical loss definitions or optimization settings.

The 2026-08-31 audit reported that all six source checkpoint MD5 values matched
the locally retained release weights. This upload preserves that audit context;
it does not reassert a fresh remote checkpoint verification. SHA256/size metadata
are retained in `checkpoints.json`, without uploading the weight binaries.
