# Final Training Commands

The existing `nita_xai/models/training_manifest.json` artifacts are debug
validation outputs. They were produced with one neural epoch so the full
pipeline could be checked quickly.

Run this local command for final 5-fold training:

```powershell
& 'C:\Users\Acer\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m nita_xai.model_training --run-mode final --epochs 30 --batch-size 512
```

Expected outputs:

- `nita_xai/models/final_training_manifest.json`
- `nita_xai/models/final/<model>/fold_<n>/...`
- `nita_xai/results/tables/final_model_metrics.csv`
- `nita_xai/results/tables/final_thresholds.csv`
- `nita_xai/results/tables/final_model_comparison.csv`

If local runtime is too long, run the same command on Lightning.ai after
installing `nita_xai/requirements.txt`. Do not present the current one-epoch
MLP/LSTM outputs as final experimental results.
