"""Data consolidation, split synchronization, and EWMA validation."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from traffic_incident_xai import config


@dataclass(frozen=True)
class StreamSplit:
    """Synchronized train/test window metadata for one incident stream."""

    stream_id: int
    fold: int
    source_file: str
    station_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_rows: int
    test_rows: int
    incident_rows_in_test: int


def ensure_directories() -> None:
    """Create isolated output folders."""

    for path in (
        config.PROCESSED_DIR,
        config.MODELS_DIR,
        config.EXPLANATIONS_DIR,
        config.REPORTS_DIR,
        config.NOTEBOOKS_DIR,
        config.TABLES_DIR,
        config.FIGURES_DIR,
        config.XAI_FIGURES_DIR,
        config.ALARM_CARDS_DIR,
        config.FINAL_PREDICTIONS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def discover_csv_files(search_dirs: Iterable[Path]) -> dict[int, Path]:
    """Find incident CSVs keyed by incident stream id."""

    files: dict[int, Path] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("incident_*.csv")):
            match = re.search(r"incident_(\d+)\.csv$", path.name)
            if match:
                files[int(match.group(1))] = path
    return files


def load_incident_csv(path: Path, stream_id: int) -> pd.DataFrame:
    """Load and normalize one incident CSV."""

    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["stream_id"] = stream_id
    frame["source_file"] = path.name
    return frame


def apply_ewma(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Apply EWMA smoothing to raw speed and occupancy columns."""

    output = frame.copy()
    output["speed_smoothed"] = (
        output["speed"].ewm(alpha=alpha, adjust=False).mean()
    )
    output["occ_smoothed"] = output["occ"].ewm(alpha=alpha, adjust=False).mean()
    return output


def validate_ewma(frame: pd.DataFrame, alpha: float) -> dict[str, float | bool]:
    """Validate stored smoothed columns against EWMA(alpha)."""

    expected = apply_ewma(frame, alpha)
    report: dict[str, float | bool] = {}
    for column in config.FEATURE_COLUMNS:
        max_error = float(
            np.nanmax(np.abs(frame[column].to_numpy() - expected[column].to_numpy()))
        )
        report[f"{column}_max_abs_error"] = max_error
        report[f"{column}_valid"] = bool(max_error < 1e-8)
    report["alpha"] = alpha
    report["valid"] = all(
        bool(report[f"{column}_valid"]) for column in config.FEATURE_COLUMNS
    )
    return report


def synchronized_windows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the strict 240-hour train and 12-hour test windows."""

    required_rows = config.TRAIN_ROWS + config.TEST_ROWS
    if len(frame) < required_rows:
        raise ValueError(
            f"Need at least {required_rows} rows, found {len(frame)} in "
            f"{frame['source_file'].iloc[0]}."
        )
    train_frame = frame.iloc[: config.TRAIN_ROWS].copy()
    test_frame = frame.iloc[
        config.TRAIN_ROWS : config.TRAIN_ROWS + config.TEST_ROWS
    ].copy()
    return train_frame, test_frame


def assign_folds(stream_ids: list[int]) -> dict[int, int]:
    """Assign stream ids to reproducible 5-fold cross-validation folds."""

    folds: dict[int, int] = {}
    rng = np.random.default_rng(config.RANDOM_STATE)
    stream_array = np.array(sorted(stream_ids), dtype=int)
    rng.shuffle(stream_array)
    fold_arrays = np.array_split(stream_array, config.N_FOLDS)
    for fold_index, fold_streams in enumerate(fold_arrays, 1):
        for stream_id in fold_streams:
            folds[int(stream_id)] = fold_index
    return folds


def consolidate_dataset(
    stream_ids: Iterable[int] = config.STREAM_IDS,
    output_path: Path = config.PROCESSED_DIR / "training_dataset.csv",
) -> pd.DataFrame:
    """Build the unified training dataset from individual smoothed CSV files."""

    ensure_directories()
    raw_files = discover_csv_files(config.RAW_SEARCH_DIRS)
    smoothed_files = discover_csv_files(config.SMOOTHED_SEARCH_DIRS)
    selected_ids = [
        stream_id for stream_id in stream_ids if stream_id in smoothed_files
    ]
    folds = assign_folds(selected_ids)
    rows: list[pd.DataFrame] = []
    metadata: list[StreamSplit] = []
    smoothing_reports: list[dict[str, object]] = []

    for stream_id in selected_ids:
        frame = load_incident_csv(smoothed_files[stream_id], stream_id)
        if not set(config.FEATURE_COLUMNS).issubset(frame.columns):
            if stream_id not in raw_files:
                raise FileNotFoundError(f"No raw CSV available for stream {stream_id}.")
            raw_frame = load_incident_csv(raw_files[stream_id], stream_id)
            frame = apply_ewma(raw_frame, config.EWMA_ALPHA)
        smoothing_report = validate_ewma(frame, config.EWMA_ALPHA)
        smoothing_report.update(
            {"stream_id": stream_id, "source_file": str(smoothed_files[stream_id])}
        )
        smoothing_reports.append(smoothing_report)

        train_frame, test_frame = synchronized_windows(frame)
        train_frame["split"] = "train"
        test_frame["split"] = "test"
        fold = folds[stream_id]
        train_frame["fold"] = fold
        test_frame["fold"] = fold
        train_frame["row_in_split"] = np.arange(len(train_frame))
        test_frame["row_in_split"] = np.arange(len(test_frame))
        rows.extend([train_frame, test_frame])
        metadata.append(
            StreamSplit(
                stream_id=stream_id,
                fold=fold,
                source_file=str(smoothed_files[stream_id]),
                station_id=int(frame["station_id"].iloc[0]),
                train_start=str(train_frame["timestamp"].iloc[0]),
                train_end=str(train_frame["timestamp"].iloc[-1]),
                test_start=str(test_frame["timestamp"].iloc[0]),
                test_end=str(test_frame["timestamp"].iloc[-1]),
                train_rows=len(train_frame),
                test_rows=len(test_frame),
                incident_rows_in_test=int(test_frame["is_incident"].sum()),
            )
        )

    if not rows:
        raise FileNotFoundError("No smoothed incident CSV files were discovered.")

    dataset = pd.concat(rows, ignore_index=True)
    dataset.insert(0, "row_id", np.arange(len(dataset)))
    dataset.to_csv(output_path, index=False)
    pd.DataFrame(asdict(item) for item in metadata).to_csv(
        config.PROCESSED_DIR / "split_metadata.csv",
        index=False,
    )
    pd.DataFrame(smoothing_reports).to_csv(
        config.PROCESSED_DIR / "ewma_validation.csv",
        index=False,
    )
    discovery_report = {
        "raw_search_dirs": [str(path) for path in config.RAW_SEARCH_DIRS],
        "smoothed_search_dirs": [str(path) for path in config.SMOOTHED_SEARCH_DIRS],
        "raw_files_found": len(raw_files),
        "smoothed_files_found": len(smoothed_files),
        "streams_used": len(selected_ids),
        "requested_raw_processed_dirs_present": {
            "data/raw": (config.PROJECT_ROOT / "data" / "raw").exists(),
            "data/processed": (config.PROJECT_ROOT / "data" / "processed").exists(),
        },
    }
    (config.PROCESSED_DIR / "discovery_report.json").write_text(
        json.dumps(discovery_report, indent=2),
        encoding="utf-8",
    )
    return dataset


def main() -> None:
    """CLI entry point for data consolidation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-streams", type=int, default=None)
    args = parser.parse_args()
    stream_ids = list(config.STREAM_IDS)
    if args.max_streams is not None:
        stream_ids = stream_ids[: args.max_streams]
    dataset = consolidate_dataset(stream_ids=stream_ids)
    print(f"Wrote {len(dataset):,} rows to {config.PROCESSED_DIR}.")


if __name__ == "__main__":
    main()
