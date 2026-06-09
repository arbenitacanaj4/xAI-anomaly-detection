"""Central configuration for the NITA XAI pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NITA_ROOT = PROJECT_ROOT / "nita_xai"

RAW_SEARCH_DIRS = (
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "incident_files" / "raw",
)
SMOOTHED_SEARCH_DIRS = (
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "incident_files" / "smoothed",
)

DATA_DIR = NITA_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = NITA_ROOT / "models"
EXPLANATIONS_DIR = NITA_ROOT / "explanations"
REPORTS_DIR = NITA_ROOT / "reports"
NOTEBOOKS_DIR = NITA_ROOT / "notebooks"
RESULTS_DIR = NITA_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
XAI_FIGURES_DIR = FIGURES_DIR / "xai"
ALARM_CARDS_DIR = RESULTS_DIR / "alarm_cards"

DEBUG_RUN_MODE = "debug"
FINAL_RUN_MODE = "final"
RUN_MODES = (DEBUG_RUN_MODE, FINAL_RUN_MODE)

DEBUG_MANIFEST_PATH = MODELS_DIR / "training_manifest.json"
FINAL_MANIFEST_PATH = MODELS_DIR / "final_training_manifest.json"
FINAL_MODEL_METRICS_PATH = TABLES_DIR / "final_model_metrics.csv"
FINAL_THRESHOLDS_PATH = TABLES_DIR / "final_thresholds.csv"
FINAL_MODEL_COMPARISON_PATH = TABLES_DIR / "final_model_comparison.csv"
FINAL_PREDICTIONS_DIR = RESULTS_DIR / "predictions"

XAI_LOCAL_EXPLANATIONS_PATH = TABLES_DIR / "xai_local_explanations.csv"
XAI_METHOD_AGREEMENT_PATH = TABLES_DIR / "xai_method_agreement.csv"
XAI_CROSS_MODEL_AGREEMENT_PATH = TABLES_DIR / "xai_cross_model_agreement.csv"
XAI_PATTERN_SUMMARY_PATH = TABLES_DIR / "xai_pattern_summary.csv"

INCIDENT_EPISODE_SUMMARY_PATH = TABLES_DIR / "incident_episode_summary.csv"
INCIDENT_LEVEL_DETECTION_PATH = TABLES_DIR / "incident_level_detection.csv"
THESIS_STYLE_MODEL_METRICS_PATH = TABLES_DIR / "thesis_style_model_metrics.csv"
ALARM_EPISODE_TRIAGE_PATH = TABLES_DIR / "alarm_episode_triage.csv"
FALSE_ALARM_TRIAGE_PATH = TABLES_DIR / "false_alarm_triage.csv"
ALARM_CARD_EXAMPLES_PATH = ALARM_CARDS_DIR / "alarm_card_examples.txt"

RANDOM_STATE = 42
N_FOLDS = 5
ROWS_PER_HOUR = 12
TRAIN_HOURS = 240
TEST_HOURS = 12
TRAIN_ROWS = TRAIN_HOURS * ROWS_PER_HOUR
TEST_ROWS = TEST_HOURS * ROWS_PER_HOUR
LOOKBACK = 12
EWMA_ALPHA = 0.15
FEATURE_COLUMNS = ("speed_smoothed", "occ_smoothed")
RAW_FEATURE_COLUMNS = ("speed", "occ")
TARGET_COLUMNS = ("speed_smoothed", "occ_smoothed")
THRESHOLD_QUANTILE = 0.95
TARGET_RECALL = 0.90
CALIBRATION_FRACTION = 0.20

DEBUG_EPOCHS = 1
FINAL_MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

RF_PARAMS = {
    "n_estimators": 120,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}
XGB_PARAMS = {
    "n_estimators": 160,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

STREAM_IDS = (
    9, 13, 16, 14, 17, 20, 23, 50, 52, 53, 54, 55, 56, 60, 63, 67, 68, 69,
    74, 78, 80, 84, 88, 90, 91, 92, 94, 95, 96, 98, 100, 101, 102, 103,
    106, 108, 109, 110, 113, 116, 117, 118, 120, 123, 127, 128, 129, 130,
    131, 132, 133, 136, 138, 142, 145, 151, 152, 153, 154, 155, 159, 161,
    162, 166, 170, 171, 172, 173, 174, 176, 177, 180, 181, 184, 185, 186,
    187, 188, 193, 197, 198, 199, 201, 202, 204, 205, 213, 216, 217, 218,
    219, 220, 221, 222, 224, 226, 227, 231, 233, 236, 238, 240, 241, 242,
    243, 245, 251, 254, 255, 259, 262, 263, 264, 265, 267, 270, 271, 273,
    274, 276, 284, 287, 289, 293, 296, 297, 304, 306, 307, 309, 317, 322,
    323, 326, 340, 341, 342, 343, 344, 345, 347, 352, 354, 356, 357, 360,
    362, 371, 373, 381, 383, 387, 390, 393, 394, 395, 398, 401, 403, 405,
    407, 408, 410, 414, 415, 416, 420, 429, 433, 435, 440, 441, 442, 448,
    450,
)
