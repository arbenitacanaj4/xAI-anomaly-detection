# XAI-Based Alarm Classification for High-Recall Traffic Incident Detection

This repository contains my Project Laboratory work at the Budapest University of Technology and Economics.

The project studies how Explainable Artificial Intelligence can add context to traffic incident alarms. Instead of treating a model output as only an alarm or no-alarm decision, the pipeline groups consecutive detections into alarm episodes and examines the traffic evidence behind each episode.

The aim is not to use XAI to prove the physical cause of an incident. The explanations are used to show which recent speed and occupancy patterns influenced the models, giving an operator more information when deciding which alarms deserve attention.

The full project report is available in [`paper.pdf`](paper.pdf).

## Problem

Traffic incident detection systems are often designed for high recall because missing an incident can be costly. A consequence is that some alarms will occur outside the labeled incident period.

A binary alarm alone does not explain whether the model reacted to a clear recent traffic change or to weaker, less consistent evidence.

This project investigates whether model explanations and temporal traffic context can provide a more useful description of an alarm.

The main research question was:

> How can XAI methods and temporal traffic context be combined to provide traffic operators with useful supporting information for judging incident alarms?

## Dataset

The experiments use prepared traffic incident-window data based on the framework of previous work at BME.

The processed dataset contains:

- 175 selected incident streams
- 240 hours of incident-free baseline data per stream
- 12 hours of test data per stream
- 5-minute sampling intervals
- 529,200 rows in total
- smoothed speed and occupancy as model inputs
- EWMA smoothing with `alpha = 0.15`

Each stream is kept intact during cross-validation. The 175 streams are divided into five folds of 35 streams so that traffic from a held-out stream does not appear in the corresponding training set.

The source traffic data and large generated model files are not stored in this repository.

## Approach

The models are used as anomaly detectors rather than standard supervised incident classifiers.

They learn normal traffic behaviour from incident-free baseline data. During testing, the difference between predicted and observed traffic conditions is converted into an anomaly score. Scores above a calibrated threshold produce an alarm.

A 60-minute lookback window is used, consisting of 12 five-minute observations of:

- `speed_smoothed`
- `occ_smoothed`

Four model families are compared:

| Model | Input | Explanation |
| --- | --- | --- |
| Random Forest | Flattened temporal window | TreeSHAP, LIME-style local surrogate |
| XGBoost | Flattened temporal window | TreeSHAP, LIME-style local surrogate |
| MLP | Flattened temporal window | DeepSHAP-style attribution, LIME-style local surrogate |
| LSTM | Sequential temporal window | Sequence DeepSHAP-style attribution |

## Alarm episodes

The models produce one prediction every five minutes, but several consecutive positive predictions usually represent the same operational alarm.

The pipeline therefore groups consecutive positive timestamps into **alarm episodes**.

One representative timestamp from each episode is selected for explanation. The triage layer combines model output with information such as:

- anomaly score and threshold margin
- alarm duration
- number of models alarming
- recent attribution share
- explanation concentration
- agreement between explanation methods
- agreement between different model families

This produces an evidence profile rather than automatically labeling an alarm as true or false.

## Results

Two evaluation views are used.

Timestamp-level evaluation measures precision, recall and F1-score for individual five-minute predictions. Incident-level evaluation measures whether an incident was detected at least once, the false alarm rate outside labeled incident periods, and mean time to detection.

The final incident-level results reported in the project were:

| Model | Detection Rate | False Alarm Rate | Mean Time to Detection |
| --- | ---: | ---: | ---: |
| Random Forest | 0.994 | 0.0583 | 0.686 min |
| XGBoost | 1.000 | 0.0579 | 1.086 min |
| MLP | 0.983 | 0.0580 | 0.724 min |
| LSTM | 0.994 | 0.0499 | 0.544 min |

The XAI analysis showed that temporal information was more informative than simply asking whether speed or occupancy was more important.

For example, the mean share of attribution assigned to the most recent 0–15 minutes was:

| Model / explanation | Recent attribution share |
| --- | ---: |
| Random Forest / TreeSHAP | 0.987 |
| XGBoost / TreeSHAP | 0.967 |
| LSTM / Sequence DeepSHAP | 0.977 |

The LIME-style explanations were generally more distributed across the full 60-minute lookback window. This difference between explanation methods became part of the analysis rather than treating one explanation as ground truth.

## Project structure

```text
traffic_incident_xai/
├── config.py
├── data_pipeline.py
├── model_training.py
├── triage.py
├── xai_alarm_episodes.py
├── interpreter.py
└── report_outputs.py

notebooks/
├── 01_data_label_check.ipynb
├── 02_model_training_tuning.ipynb
├── 03_lstm_sequence_model.ipynb
├── 04_xai_model_explanations.ipynb
├── 05_alarm_triage_layer.ipynb
└── 06_report_outputs.ipynb

tests/
└── test_core_behaviors.py

results/
└── figures/

paper.pdf
paper.tex
requirements.txt
```

Large datasets, trained models, prediction tables and generated CSV outputs are kept outside version control.

## Setup

The project has been tested with Python 3.11.

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The data-processing and training commands require the prepared incident CSV files to be available in the configured data directories.

## Running the pipeline

Build and validate the consolidated dataset:

```bash
python -m traffic_incident_xai.data_pipeline
```

Run the final five-fold training:

```bash
python -m traffic_incident_xai.model_training --run-mode final --folds 1 2 3 4 5 --epochs 30 --batch-size 512
```

Build alarm episodes and triage outputs:

```bash
python -m traffic_incident_xai.triage
```

Check XAI coverage:

```bash
python -m traffic_incident_xai.xai_alarm_episodes --status
```

Compute missing alarm-episode explanations:

```bash
python -m traffic_incident_xai.xai_alarm_episodes --compute-missing
```

Build the XAI summary and agreement tables:

```bash
python -m traffic_incident_xai.xai_alarm_episodes --build-tables
```

Generate the report figures:

```bash
python -c "from traffic_incident_xai import report_outputs as r; r.build_all_report_figures()"
```

## Tests

The repository includes a small synthetic test suite for the core pipeline behaviour, including stream boundaries, temporal window construction, fold assignment, anomaly scoring, threshold calibration and alarm episode grouping.

Run it with:

```bash
python -m unittest discover -s tests -v
```

## Limitations

The models only receive smoothed speed and occupancy, so the explanations cannot provide evidence from variables that were not part of the model input.

The LSTM is explained using sequence attribution rather than the LIME-style local surrogate used for the flattened models, since independently perturbing values in an ordered sequence can produce unrealistic temporal inputs.

XAI is also calculated for one representative timestamp per alarm episode. This provides a compact explanation of the alarm but does not show how feature attribution changes throughout the entire episode.

Most importantly, the explanations describe model behaviour. They should not be interpreted as proof of the physical cause of a traffic incident.

## Project report

A detailed discussion of the experimental setup, evaluation, XAI analysis and operator-facing alarm evidence is available in:

[`paper.pdf`](paper.pdf)

The project was completed as part of the BSc Computer Engineering Project Laboratory at the Budapest University of Technology and Economics.
