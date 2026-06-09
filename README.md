# NITA XAI: Traffic Incident Alarm Triage

This folder contains the final project implementation for an XAI-based traffic incident alarm triage experiment.

The project uses prepared traffic incident-window data from the referenced thesis as the data and protocol foundation. It does not reproduce the whole thesis pipeline. The focus here is narrower:

> build high-recall traffic anomaly alarms, then use XAI and temporal context to help an operator judge the alarm evidence.

XAI is used to explain model behavior, not to prove the physical cause of an incident.

## What the Pipeline Does

1. Consolidates 175 selected incident streams into one processed dataset.
2. Keeps the protocol of 240 hours of incident-free baseline data and 12 hours of test data per stream.
3. Splits whole streams into 5 folds, with 35 streams per fold.
4. Trains four model families:
   - Random Forest
   - XGBoost
   - MLP
   - LSTM
5. Produces anomaly scores, calibrated high-recall thresholds, and timestamp-level predictions.
6. Groups consecutive positive predictions into alarm episodes.
7. Computes XAI for alarm episode representative rows across all folds.
8. Summarizes alarm evidence for report figures and operator-facing triage.

## Main Data Assumptions

- One stream corresponds to one selected incident-window timeline.
- Each stream has 2880 baseline rows and 144 test rows.
- Sampling interval is 5 minutes.
- Model inputs are:
  - `speed_smoothed`
  - `occ_smoothed`
- EWMA smoothing was validated with alpha `0.15`.
- Labels are used for offline evaluation, not as information available to the operator in real time.

## Important Files

| File | Purpose |
|---|---|
| `config.py` | Central paths and project settings. |
| `data_pipeline.py` | Loads incident CSVs, validates smoothing, builds `training_dataset.csv`, and assigns folds. |
| `model_training.py` | Trains RF, XGBoost, MLP, and LSTM models and saves final model outputs. |
| `triage.py` | Combines predictions, groups timestamp alarms into episodes, and creates operator triage tables. |
| `xai_alarm_episodes.py` | Computes all-episode XAI explanations and agreement metrics. |
| `report_outputs.py` | Generates report-ready figures used by notebook 06 and the report. |
| `interpreter.py` | Prototype diagnostic wording from attribution patterns. |

## Important Folders

| Folder | Purpose |
|---|---|
| `data/processed/` | Consolidated dataset and data validation outputs. |
| `models/final/` | Final trained model files and fold score CSVs. |
| `results/predictions/` | Combined prediction tables used by XAI and triage. |
| `results/tables/` | Final metrics, thresholds, XAI tables, and triage tables. |
| `results/figures/report/` | Report-ready PNG figures and figure manifest. |
| `results/alarm_cards/` | Example operator-facing alarm card text. |
| `notebooks/` | Restartable notebooks for explanation, inspection, and final report outputs. |

## Final Outputs to Know

Training and evaluation:

- `results/tables/final_model_metrics.csv`
- `results/tables/final_model_comparison.csv`
- `results/tables/final_thresholds.csv`
- `results/tables/thesis_style_model_metrics.csv`
- `models/final_training_manifest.json`

Predictions and triage:

- `results/predictions/all_model_predictions.csv`
- `results/tables/incident_level_detection.csv`
- `results/tables/incident_episode_summary.csv`
- `results/tables/alarm_episode_triage.csv`
- `results/tables/false_alarm_triage.csv`

XAI:

- `results/tables/xai_alarm_episode_targets.csv`
- `results/tables/xai_alarm_episode_coverage.csv`
- `results/tables/xai_local_explanations_all_alarm_episodes.csv`
- `results/tables/xai_alarm_evidence_metrics.csv`
- `results/tables/xai_method_agreement_all_alarm_episodes.csv`
- `results/tables/xai_cross_model_agreement_all_alarm_episodes.csv`

Report figures:

- `results/figures/report/report_figure_manifest.csv`
- `results/figures/report/*.png`

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_label_check.ipynb` | Explains the data source, 175 streams, windows, labels, and incident durations. |
| `02_model_training_tuning.ipynb` | Explains model families, thresholds, debug vs final training, and final metrics. |
| `03_lstm_sequence_model.ipynb` | Explains LSTM sequence construction and stream-boundary safety. |
| `04_xai_model_explanations.ipynb` | Explains SHAP/LIME/sequence XAI and agreement outputs. |
| `05_alarm_triage_layer.ipynb` | Explains timestamp predictions vs alarm episodes and operator triage. |
| `06_report_outputs.ipynb` | Generates and displays final report-ready figures. |

## Common Commands

Run data consolidation and validation:

```bash
python -m nita_xai.data_pipeline
```

Run final training locally:

```bash
python -m nita_xai.model_training --run-mode final --models random_forest xgboost mlp lstm --folds 1 2 3 4 5 --epochs 30 --early-stopping
```

Build episode-level triage outputs:

```bash
python -m nita_xai.triage
```

Build all-alarm XAI tables:

```bash
python -m nita_xai.xai_alarm_episodes
```

Generate report figures:

```bash
python -c "from nita_xai import report_outputs as r; r.build_all_report_figures()"
```

## Notes

- The final manifest is `models/final_training_manifest.json`.
- The project should use final outputs under `models/final/` and `results/`.
- The old intermediate `explanations/` folder was removed because final XAI tables are stored under `results/tables/`.
- False alarms are not treated as useless. They are analyzed as alarm episodes so the operator can judge their evidence quality.
