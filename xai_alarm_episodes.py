"""XAI coverage for all alarm episode representative rows."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge

from nita_xai import config
from nita_xai.model_training import (
    StationLstmRegressor,
    TabularMlpRegressor,
    flatten_windows,
    load_dataset,
    make_windows,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


TARGETS_PATH = config.TABLES_DIR / "xai_alarm_episode_targets.csv"
COVERAGE_PATH = config.TABLES_DIR / "xai_alarm_episode_coverage.csv"
LOCAL_PATH = config.TABLES_DIR / "xai_local_explanations_all_alarm_episodes.csv"
EVIDENCE_PATH = config.TABLES_DIR / "xai_alarm_evidence_metrics.csv"
METHOD_AGREEMENT_PATH = (
    config.TABLES_DIR / "xai_method_agreement_all_alarm_episodes.csv"
)
CROSS_MODEL_AGREEMENT_PATH = (
    config.TABLES_DIR / "xai_cross_model_agreement_all_alarm_episodes.csv"
)


TABULAR_METHODS = {
    "random_forest": ("treeshap", "lime"),
    "xgboost": ("treeshap", "lime"),
    "mlp": ("deepshap", "lime"),
}
SEQUENCE_METHODS = {"lstm": ("sequence_deepshap",)}
ALL_METHODS = {**TABULAR_METHODS, **SEQUENCE_METHODS}


def feature_names() -> list[str]:
    """Return flattened tabular lookback feature names."""

    names: list[str] = []
    for lag in range(config.LOOKBACK, 0, -1):
        for feature in config.FEATURE_COLUMNS:
            names.append(f"{feature}_t_minus_{lag}")
    return names


def sample_rows(array: np.ndarray, sample_size: int) -> np.ndarray:
    """Deterministically sample rows for tractable SHAP computation."""

    if len(array) <= sample_size:
        return array
    rng = np.random.default_rng(config.RANDOM_STATE)
    indices = rng.choice(len(array), size=sample_size, replace=False)
    return array[np.sort(indices)]


def model_fold_dir(model_name: str, fold: int) -> Path:
    """Return the active final model directory for a model/fold."""

    final_dir = config.MODELS_DIR / config.FINAL_RUN_MODE / model_name / f"fold_{fold}"
    if final_dir.exists():
        return final_dir
    return config.MODELS_DIR / model_name / f"fold_{fold}"


@lru_cache(maxsize=None)
def load_window_data(fold: int) -> dict[str, Any]:
    """Load fold-specific windows, targets, stations, and metadata."""

    dataset = load_dataset()
    sequences, targets, stations, metadata = make_windows(dataset)
    train_mask = (metadata["split"] == "train") & (metadata["fold"] != fold)
    test_mask = (metadata["split"] == "test") & (metadata["fold"] == fold)
    return {
        "train_sequences": sequences[train_mask.to_numpy()],
        "test_sequences": sequences[test_mask.to_numpy()],
        "train_targets": targets[train_mask.to_numpy()],
        "test_targets": targets[test_mask.to_numpy()],
        "train_stations": stations[train_mask.to_numpy()],
        "test_stations": stations[test_mask.to_numpy()],
        "test_metadata": metadata.loc[test_mask].reset_index(drop=True),
    }


def aggregate_output_shap(values: Any) -> np.ndarray:
    """Convert per-output SHAP values to one attribution matrix."""

    if isinstance(values, list):
        stacked = np.stack(values, axis=0)
        return np.mean(np.abs(stacked), axis=0)
    array = np.asarray(values)
    if array.ndim == 3:
        return np.mean(np.abs(array), axis=-1)
    return np.abs(array)


class LstmSequenceWrapper(torch.nn.Module if torch is not None else object):  # type: ignore[misc]
    """Wrap LSTM with fixed station IDs so SHAP attributes time features."""

    def __init__(self, model: StationLstmRegressor, station_ids: "torch.Tensor") -> None:
        if torch is not None:
            super().__init__()
        self.model = model
        self.station_ids = station_ids

    def forward(self, sequence: "torch.Tensor") -> "torch.Tensor":
        """Forward with repeated or sliced station IDs."""

        station_ids = self.station_ids[: sequence.size(0)]
        if len(station_ids) < sequence.size(0):
            station_ids = station_ids.repeat(sequence.size(0))[: sequence.size(0)]
        return self.model(sequence, station_ids)


def attribution_direction(value: float) -> str:
    """Convert a signed attribution into an operator-readable direction."""

    if value > 0:
        return "supports_alarm"
    if value < 0:
        return "supports_normal"
    return "neutral"


def agreement_level(score: float | None) -> str:
    """Classify explanation agreement score."""

    if score is None or np.isnan(score):
        return "not_available"
    if score >= 0.70:
        return "consistent"
    if score >= 0.40:
        return "partially_consistent"
    return "conflicting"


def explanation_summary(level: str) -> str:
    """Plain-language agreement summary."""

    if level == "consistent":
        return "The explanations emphasize similar traffic evidence."
    if level == "partially_consistent":
        return "The explanations share some evidence but differ in emphasis."
    if level == "conflicting":
        return "The explanations point to different supporting evidence."
    return "Agreement could not be assessed for this method pair."


def top_feature_overlap(left: pd.DataFrame, right: pd.DataFrame, k: int = 5) -> float:
    """Compute top-k feature overlap between two explanation slices."""

    left_features = set(left.nsmallest(k, "rank")["feature"])
    right_features = set(right.nsmallest(k, "rank")["feature"])
    if not left_features or not right_features:
        return float("nan")
    return len(left_features & right_features) / k


def direction_agreement(left: pd.DataFrame, right: pd.DataFrame, k: int = 5) -> float:
    """Compute direction agreement for shared top features."""

    left_top = left.nsmallest(k, "rank").set_index("feature")
    right_top = right.nsmallest(k, "rank").set_index("feature")
    shared = sorted(set(left_top.index) & set(right_top.index))
    if not shared:
        return float("nan")
    if (
        "not_available_abs_only" in set(left_top.loc[shared, "direction"])
        or "not_available_abs_only" in set(right_top.loc[shared, "direction"])
    ):
        return float("nan")
    matches = [
        left_top.loc[feature, "direction"] == right_top.loc[feature, "direction"]
        for feature in shared
    ]
    return float(np.mean(matches))


def expected_method_name(model_name: str, method: str) -> str:
    """Return a stable method label for all-alarm XAI rows."""

    if method == "treeshap":
        return "treeshap"
    if method == "deepshap":
        return "deepshap"
    if method == "sequence_deepshap":
        return "sequence_deepshap"
    return "lime_style_local_surrogate"


def feature_family(feature: str) -> str:
    """Map lagged feature names to traffic signal families."""

    if "speed" in feature:
        return "speed"
    if "occ" in feature:
        return "occupancy"
    return "other"


def feature_lag(feature: str) -> int | None:
    """Extract lag steps from a feature name."""

    match = re.search(r"t_minus_(\d+)", feature)
    if match:
        return int(match.group(1))
    return None


def recency_bucket(lag_steps: int | None) -> str:
    """Group lagged features into operator-friendly time buckets."""

    if lag_steps is None:
        return "not_available"
    minutes = lag_steps * 5
    if minutes <= 15:
        return "recent_0_15_min"
    if minutes <= 30:
        return "mid_15_30_min"
    return "older_30_60_min"


def enrich_xai_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Add traffic signal and recency columns to local XAI rows."""

    enriched = frame.copy()
    enriched["traffic_signal"] = enriched["feature"].map(feature_family)
    enriched["lag_steps"] = enriched["feature"].map(feature_lag)
    enriched["minutes_before_alarm"] = enriched["lag_steps"].map(
        lambda value: value * 5 if value is not None else np.nan
    )
    enriched["recency_bucket"] = enriched["lag_steps"].map(recency_bucket)
    return enriched


def build_targets() -> pd.DataFrame:
    """Build XAI target rows from every alarm episode representative row."""

    episodes = pd.read_csv(config.ALARM_EPISODE_TRIAGE_PATH)
    predictions = pd.read_csv(config.FINAL_PREDICTIONS_DIR / "all_model_predictions.csv")
    manifest = pd.DataFrame(json.loads(config.FINAL_MANIFEST_PATH.read_text()))
    prediction_columns = [
        "row_id",
        "model",
        "fold",
        "score",
        "normalized_score",
        "threshold",
        "threshold_margin",
        "prediction",
        "true_label",
        "case_type",
    ]
    targets = episodes.merge(
        predictions[prediction_columns],
        left_on=["representative_row_id", "model", "fold"],
        right_on=["row_id", "model", "fold"],
        how="left",
    )
    targets = targets.merge(
        manifest[["model_name", "fold", "model_path", "score_path"]],
        left_on=["model", "fold"],
        right_on=["model_name", "fold"],
        how="left",
    )
    targets = targets[
        [
            "fold",
            "model",
            "stream_id",
            "source_file",
            "representative_row_id",
            "alarm_episode_id",
            "alarm_start",
            "alarm_end",
            "alarm_duration_minutes",
            "score",
            "normalized_score",
            "threshold",
            "threshold_margin",
            "prediction",
            "true_label",
            "case_type",
            "model_agreement_count",
            "reliability_level",
            "false_positive_risk_level",
            "operator_priority",
            "model_path",
            "score_path",
        ]
    ].rename(columns={"threshold_margin": "score_margin"})
    targets = targets.sort_values(["fold", "model", "stream_id", "alarm_start"])
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    targets.to_csv(TARGETS_PATH, index=False)
    return targets


def expected_cases(targets: pd.DataFrame) -> pd.DataFrame:
    """Return expected model/fold/row/method combinations."""

    rows = []
    for target in targets.itertuples(index=False):
        for method in ALL_METHODS[str(target.model)]:
            rows.append(
                {
                    "fold": int(target.fold),
                    "model": str(target.model),
                    "representative_row_id": int(target.representative_row_id),
                    "method": expected_method_name(str(target.model), method),
                }
            )
    return pd.DataFrame(rows)


def load_existing_local() -> pd.DataFrame:
    """Load all-alarm local explanations when present."""

    if LOCAL_PATH.exists():
        return pd.read_csv(LOCAL_PATH)
    return pd.DataFrame()


def coverage_table(targets: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    """Summarize XAI coverage by model/fold/method."""

    expected = expected_cases(targets)
    if local.empty:
        expected["explained"] = False
    else:
        actual = local[
            ["fold", "model", "row_id", "method"]
        ].drop_duplicates().rename(columns={"row_id": "representative_row_id"})
        expected = expected.merge(
            actual.assign(explained=True),
            on=["fold", "model", "representative_row_id", "method"],
            how="left",
        )
        expected["explained"] = expected["explained"].fillna(False)
    coverage = (
        expected.groupby(["model", "fold", "method"])
        .agg(
            target_cases=("representative_row_id", "nunique"),
            explained_cases=("explained", "sum"),
        )
        .reset_index()
    )
    coverage["missing_cases"] = (
        coverage["target_cases"] - coverage["explained_cases"]
    )
    coverage["coverage_rate"] = (
        coverage["explained_cases"] / coverage["target_cases"]
    )
    coverage["coverage_note"] = (
        "XAI is unavailable for a case when its representative_row_id, "
        "model, fold, and method are not present in "
        "xai_local_explanations_all_alarm_episodes.csv. This happens "
        "because XAI rows are only present after explicit computation."
    )
    coverage.to_csv(COVERAGE_PATH, index=False)
    return coverage


def case_indices_for_fold(fold: int, row_ids: list[int]) -> list[int]:
    """Map row ids to fold-specific test-window positions."""

    metadata = load_window_data(fold)["test_metadata"]
    row_to_index = {
        int(row_id): int(index)
        for index, row_id in enumerate(metadata["row_id"].tolist())
    }
    return [row_to_index[row_id] for row_id in row_ids if row_id in row_to_index]


def rows_from_values(
    values: np.ndarray,
    names: list[str],
    targets: pd.DataFrame,
    model_name: str,
    method: str,
) -> list[dict[str, object]]:
    """Convert attribution matrices to long-form local explanation rows."""

    rows: list[dict[str, object]] = []
    if values.ndim == 3:
        names = [
            f"{feature}_t_minus_{config.LOOKBACK - step}"
            for step in range(values.shape[1])
            for feature in names
        ]
        values = values.reshape(values.shape[0], -1)
    for case_position, target in enumerate(targets.itertuples(index=False)):
        row_values = values[case_position]
        for rank, feature_index in enumerate(np.argsort(-np.abs(row_values)), 1):
            value = float(row_values[feature_index])
            rows.append(
                {
                    "case_id": (
                        f"fold_{int(target.fold)}_{model_name}_"
                        f"{int(target.representative_row_id)}"
                    ),
                    "row_id": int(target.representative_row_id),
                    "stream_id": int(target.stream_id),
                    "source_file": str(target.source_file),
                    "timestamp": str(target.alarm_start),
                    "alarm_episode_id": int(target.alarm_episode_id),
                    "alarm_start": str(target.alarm_start),
                    "alarm_end": str(target.alarm_end),
                    "fold": int(target.fold),
                    "model": model_name,
                    "method": method,
                    "feature": names[feature_index],
                    "attribution": value,
                    "abs_attribution": abs(value),
                    "direction": attribution_direction(value),
                    "rank": rank,
                    "method_note": (
                        "All-alarm episode representative-row XAI. "
                        "Attributions explain model behavior, not physical cause."
                    ),
                }
            )
    return rows


def signed_aggregate(values: Any) -> np.ndarray:
    """Aggregate multi-output SHAP values while preserving sign."""

    if isinstance(values, list):
        return np.mean(np.stack(values, axis=0), axis=0)
    array = np.asarray(values)
    if array.ndim == 4:
        return np.mean(array, axis=-1)
    if array.ndim == 3:
        return np.mean(array, axis=-1)
    return array


def compute_tree_shap(
    model_name: str,
    fold: int,
    targets: pd.DataFrame,
    background_size: int = 40,
) -> list[dict[str, object]]:
    """Compute TreeSHAP rows for RF/XGBoost target cases."""

    data = load_window_data(fold)
    indices = case_indices_for_fold(
        fold,
        targets["representative_row_id"].astype(int).tolist(),
    )
    if not indices:
        return []
    artifact = joblib.load(model_fold_dir(model_name, fold) / "model.joblib")
    x_train = artifact["x_scaler"].transform(flatten_windows(data["train_sequences"]))
    x_test = artifact["x_scaler"].transform(flatten_windows(data["test_sequences"]))
    explain = x_test[indices]
    if model_name == "random_forest":
        explainer = shap.TreeExplainer(artifact["model"])
    else:
        explainer = shap.TreeExplainer(
            artifact["model"],
            data=sample_rows(x_train, background_size),
        )
    values = signed_aggregate(explainer.shap_values(explain, check_additivity=False))
    ordered_targets = targets[
        targets["representative_row_id"].isin(
            data["test_metadata"].iloc[indices]["row_id"].astype(int).tolist()
        )
    ].copy()
    ordered_targets["_order"] = ordered_targets["representative_row_id"].map(
        {
            int(row_id): order
            for order, row_id in enumerate(
                data["test_metadata"].iloc[indices]["row_id"].astype(int).tolist()
            )
        }
    )
    ordered_targets = ordered_targets.sort_values("_order").drop(columns="_order")
    return rows_from_values(
        values,
        feature_names(),
        ordered_targets,
        model_name,
        "treeshap",
    )


def compute_mlp_deepshap(
    fold: int,
    targets: pd.DataFrame,
    background_size: int = 40,
) -> list[dict[str, object]]:
    """Compute DeepSHAP rows for MLP target cases."""

    if torch is None:
        return []
    data = load_window_data(fold)
    indices = case_indices_for_fold(
        fold,
        targets["representative_row_id"].astype(int).tolist(),
    )
    if not indices:
        return []
    model_dir = model_fold_dir("mlp", fold)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    scalers = joblib.load(model_dir / "scalers.joblib")
    model = TabularMlpRegressor(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x_train = scalers["x_scaler"].transform(flatten_windows(data["train_sequences"]))
    x_test = scalers["x_scaler"].transform(flatten_windows(data["test_sequences"]))
    background = torch.tensor(
        sample_rows(x_train, background_size),
        dtype=torch.float32,
    )
    explain = torch.tensor(x_test[indices], dtype=torch.float32)
    explainer = shap.DeepExplainer(model, background)
    values = signed_aggregate(
        explainer.shap_values(explain, check_additivity=False)
    )
    ordered_targets = targets[
        targets["representative_row_id"].isin(
            data["test_metadata"].iloc[indices]["row_id"].astype(int).tolist()
        )
    ].copy()
    ordered_targets["_order"] = ordered_targets["representative_row_id"].map(
        {
            int(row_id): order
            for order, row_id in enumerate(
                data["test_metadata"].iloc[indices]["row_id"].astype(int).tolist()
            )
        }
    )
    ordered_targets = ordered_targets.sort_values("_order").drop(columns="_order")
    return rows_from_values(
        values,
        feature_names(),
        ordered_targets,
        "mlp",
        "deepshap",
    )


def compute_lstm_deepshap(
    fold: int,
    targets: pd.DataFrame,
    background_size: int = 20,
) -> list[dict[str, object]]:
    """Compute sequence DeepSHAP rows for LSTM target cases."""

    if torch is None:
        return []
    data = load_window_data(fold)
    indices = case_indices_for_fold(
        fold,
        targets["representative_row_id"].astype(int).tolist(),
    )
    if not indices:
        return []
    model_dir = model_fold_dir("lstm", fold)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    scalers = joblib.load(model_dir / "scalers.joblib")
    model = StationLstmRegressor(
        n_stations=int(checkpoint["n_stations"]),
        n_features=int(checkpoint["n_features"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    train_sequences = data["train_sequences"]
    test_sequences = data["test_sequences"]
    x_train = scalers["x_scaler"].transform(
        train_sequences.reshape(-1, train_sequences.shape[-1])
    ).reshape(train_sequences.shape)
    x_test = scalers["x_scaler"].transform(
        test_sequences.reshape(-1, test_sequences.shape[-1])
    ).reshape(test_sequences.shape)
    station_map = checkpoint["station_map"]
    target_stations = torch.tensor(
        [
            station_map.get(int(station), 0)
            for station in data["test_stations"][indices]
        ],
        dtype=torch.long,
    )
    background_np = sample_rows(x_train, background_size)
    explain_np = x_test[indices]
    wrapper = LstmSequenceWrapper(model, target_stations)
    background = torch.tensor(background_np, dtype=torch.float32)
    explain = torch.tensor(explain_np, dtype=torch.float32)
    explainer = shap.DeepExplainer(wrapper, background)
    values = signed_aggregate(
        explainer.shap_values(explain, check_additivity=False)
    )
    ordered_targets = targets[
        targets["representative_row_id"].isin(
            data["test_metadata"].iloc[indices]["row_id"].astype(int).tolist()
        )
    ].copy()
    ordered_targets["_order"] = ordered_targets["representative_row_id"].map(
        {
            int(row_id): order
            for order, row_id in enumerate(
                data["test_metadata"].iloc[indices]["row_id"].astype(int).tolist()
            )
        }
    )
    ordered_targets = ordered_targets.sort_values("_order").drop(columns="_order")
    return rows_from_values(
        values,
        list(config.FEATURE_COLUMNS),
        ordered_targets,
        "lstm",
        "sequence_deepshap",
    )


def compute_lime_surrogate(
    model_name: str,
    fold: int,
    targets: pd.DataFrame,
    num_samples: int = 30,
) -> list[dict[str, object]]:
    """Compute LIME-style local surrogate rows for tabular models."""

    if model_name == "lstm":
        return []
    if torch is None and model_name == "mlp":
        return []
    data = load_window_data(fold)
    indices = case_indices_for_fold(
        fold,
        targets["representative_row_id"].astype(int).tolist(),
    )
    if not indices:
        return []
    train_x = flatten_windows(data["train_sequences"])
    test_x = flatten_windows(data["test_sequences"])
    if model_name in {"random_forest", "xgboost"}:
        artifact = joblib.load(model_fold_dir(model_name, fold) / "model.joblib")
        x_train = artifact["x_scaler"].transform(train_x)
        x_test = artifact["x_scaler"].transform(test_x)

        def predict_output(array: np.ndarray, output_index: int) -> np.ndarray:
            return artifact["model"].predict(array)[:, output_index]

    else:
        model_dir = model_fold_dir("mlp", fold)
        checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
        scalers = joblib.load(model_dir / "scalers.joblib")
        model = TabularMlpRegressor(
            input_dim=int(checkpoint["input_dim"]),
            output_dim=int(checkpoint["output_dim"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x_train = scalers["x_scaler"].transform(train_x)
        x_test = scalers["x_scaler"].transform(test_x)

        def predict_output(array: np.ndarray, output_index: int) -> np.ndarray:
            with torch.no_grad():
                prediction = model(torch.tensor(array, dtype=torch.float32)).numpy()
            return prediction[:, output_index]

    names = feature_names()
    feature_scale = np.std(x_train, axis=0) + 1e-6
    rng = np.random.default_rng(config.RANDOM_STATE + fold + len(model_name))
    ordered_metadata = data["test_metadata"].iloc[indices].reset_index(drop=True)
    ordered_targets = targets[
        targets["representative_row_id"].isin(
            ordered_metadata["row_id"].astype(int).tolist()
        )
    ].copy()
    ordered_targets["_order"] = ordered_targets["representative_row_id"].map(
        {
            int(row_id): order
            for order, row_id in enumerate(ordered_metadata["row_id"].astype(int))
        }
    )
    ordered_targets = ordered_targets.sort_values("_order").drop(columns="_order")
    rows: list[dict[str, object]] = []
    for order, target in enumerate(ordered_targets.itertuples(index=False)):
        instance = x_test[indices[order]]
        perturbations = rng.normal(
            loc=instance,
            scale=0.15 * feature_scale,
            size=(num_samples, len(names)),
        )
        perturbations[0] = instance
        distances = np.linalg.norm(
            (perturbations - instance) / feature_scale,
            axis=1,
        )
        weights = np.exp(-(distances**2) / (len(names) * 0.75))
        combined = np.zeros(len(names), dtype=float)
        for output_index in range(len(config.FEATURE_COLUMNS)):
            local_model = Ridge(alpha=1.0)
            local_model.fit(
                perturbations,
                predict_output(perturbations, output_index),
                sample_weight=weights,
            )
            combined += local_model.coef_ / len(config.FEATURE_COLUMNS)
        rows.extend(
            rows_from_values(
                combined.reshape(1, -1),
                names,
                pd.DataFrame([target._asdict()]),
                model_name,
                "lime_style_local_surrogate",
            )
        )
    return rows


def append_rows(rows: list[dict[str, object]]) -> None:
    """Append local XAI rows to disk safely."""

    if not rows:
        return
    frame = enrich_xai_rows(pd.DataFrame(rows))
    header = not LOCAL_PATH.exists()
    frame.to_csv(LOCAL_PATH, mode="a", header=header, index=False)


def missing_targets(
    targets: pd.DataFrame,
    local: pd.DataFrame,
    model_name: str,
    fold: int,
    method: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return targets missing a model/fold/method explanation."""

    subset = targets[(targets["model"] == model_name) & (targets["fold"] == fold)]
    if local.empty:
        missing = subset
    else:
        actual_ids = set(
            local[
                (local["model"] == model_name)
                & (local["fold"] == fold)
                & (local["method"] == method)
            ]["row_id"].astype(int)
        )
        missing = subset[
            ~subset["representative_row_id"].astype(int).isin(actual_ids)
        ]
    if limit is not None:
        return missing.head(limit)
    return missing


def compute_missing_xai(limit_per_group: int | None = None) -> None:
    """Compute missing all-alarm XAI rows incrementally by model/fold."""

    targets = build_targets()
    local = load_existing_local()
    coverage_table(targets, local)
    for model_name, fold in itertools.product(
        sorted(targets["model"].unique()),
        sorted(targets["fold"].unique()),
    ):
        if model_name not in ALL_METHODS:
            continue
        for method in ALL_METHODS[model_name]:
            label = expected_method_name(model_name, method)
            group_targets = missing_targets(
                targets,
                local,
                model_name,
                int(fold),
                label,
                limit=limit_per_group,
            )
            if group_targets.empty:
                continue
            print(
                f"Computing {label}: model={model_name}, fold={fold}, "
                f"cases={len(group_targets)}"
            )
            if method == "treeshap":
                rows = compute_tree_shap(model_name, int(fold), group_targets)
            elif method == "deepshap":
                rows = compute_mlp_deepshap(int(fold), group_targets)
            elif method == "sequence_deepshap":
                rows = compute_lstm_deepshap(int(fold), group_targets)
            else:
                rows = compute_lime_surrogate(model_name, int(fold), group_targets)
            append_rows(rows)
            local = load_existing_local()
            coverage_table(targets, local)
            build_evidence_and_agreement_tables(local)


def normalized_entropy(values: pd.Series) -> float:
    """Compute normalized entropy of absolute attribution magnitudes."""

    array = values.to_numpy(dtype=float)
    total = float(array.sum())
    if total <= 0 or len(array) <= 1:
        return 0.0
    probabilities = array / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return entropy / math.log(len(array))


def build_evidence_metrics(local: pd.DataFrame) -> pd.DataFrame:
    """Aggregate local XAI rows into per-case evidence-quality metrics."""

    local = enrich_xai_rows(local)
    rows = []
    for keys, group in local.groupby(
        ["case_id", "row_id", "fold", "model", "method"],
    ):
        case_id, row_id, fold, model_name, method = keys
        top10 = group.nsmallest(10, "rank")
        top3 = group.nsmallest(3, "rank")
        total = float(top10["abs_attribution"].sum())
        if total <= 0:
            continue
        rows.append(
            {
                "case_id": case_id,
                "row_id": int(row_id),
                "fold": int(fold),
                "model": model_name,
                "method": method,
                "top_10_features": ", ".join(top10["feature"].tolist()),
                "speed_attribution_share": float(
                    top10.loc[
                        top10["traffic_signal"] == "speed",
                        "abs_attribution",
                    ].sum()
                    / total
                ),
                "occupancy_attribution_share": float(
                    top10.loc[
                        top10["traffic_signal"] == "occupancy",
                        "abs_attribution",
                    ].sum()
                    / total
                ),
                "recent_0_15_min_share": float(
                    top10.loc[
                        top10["recency_bucket"] == "recent_0_15_min",
                        "abs_attribution",
                    ].sum()
                    / total
                ),
                "mid_15_30_min_share": float(
                    top10.loc[
                        top10["recency_bucket"] == "mid_15_30_min",
                        "abs_attribution",
                    ].sum()
                    / total
                ),
                "older_30_60_min_share": float(
                    top10.loc[
                        top10["recency_bucket"] == "older_30_60_min",
                        "abs_attribution",
                    ].sum()
                    / total
                ),
                "top3_attribution_concentration": float(
                    top3["abs_attribution"].sum() / total
                ),
                "explanation_entropy": normalized_entropy(
                    top10["abs_attribution"]
                ),
                "explanation_diffuseness": normalized_entropy(
                    top10["abs_attribution"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_method_agreement(local: pd.DataFrame) -> pd.DataFrame:
    """Build SHAP/LIME agreement for all alarm episode rows."""

    rows = []
    for (case_id, model_name), group in local.groupby(["case_id", "model"]):
        shap_group = group[group["method"].str.contains("shap")]
        lime_group = group[group["method"] == "lime_style_local_surrogate"]
        if shap_group.empty or lime_group.empty:
            continue
        overlap = top_feature_overlap(shap_group, lime_group)
        direction = direction_agreement(shap_group, lime_group)
        score = overlap if np.isnan(direction) else float(np.mean([overlap, direction]))
        level = agreement_level(score)
        rows.append(
            {
                "case_id": case_id,
                "row_id": int(group["row_id"].iloc[0]),
                "fold": int(group["fold"].iloc[0]),
                "model": model_name,
                "top_feature_overlap_score": overlap,
                "direction_agreement_score": direction,
                "explanation_agreement_level": level,
                "summary": explanation_summary(level),
            }
        )
    return pd.DataFrame(rows)


def build_cross_model_agreement(local: pd.DataFrame) -> pd.DataFrame:
    """Build cross-model agreement for rows where multiple models alarmed."""

    rows = []
    shap_like = local[local["method"].str.contains("shap")].copy()
    for row_id, group in shap_like.groupby("row_id"):
        models = sorted(group["model"].unique())
        for left_model, right_model in itertools.combinations(models, 2):
            left = group[group["model"] == left_model]
            right = group[group["model"] == right_model]
            overlap = top_feature_overlap(left, right)
            direction = direction_agreement(left, right)
            score = overlap if np.isnan(direction) else float(np.mean([overlap, direction]))
            level = agreement_level(score)
            rows.append(
                {
                    "row_id": int(row_id),
                    "left_model": left_model,
                    "right_model": right_model,
                    "top_feature_overlap_score": overlap,
                    "direction_agreement_score": direction,
                    "explanation_agreement_level": level,
                    "summary": explanation_summary(level),
                }
            )
    return pd.DataFrame(rows)


def build_evidence_and_agreement_tables(local: pd.DataFrame | None = None) -> None:
    """Write all-alarm evidence metrics and agreement tables."""

    if local is None:
        local = load_existing_local()
    if local.empty:
        return
    local = enrich_xai_rows(local)
    local.to_csv(LOCAL_PATH, index=False)
    build_evidence_metrics(local).to_csv(EVIDENCE_PATH, index=False)
    build_method_agreement(local).to_csv(METHOD_AGREEMENT_PATH, index=False)
    build_cross_model_agreement(local).to_csv(
        CROSS_MODEL_AGREEMENT_PATH,
        index=False,
    )
    targets = pd.read_csv(TARGETS_PATH) if TARGETS_PATH.exists() else build_targets()
    coverage_table(targets, local)


def print_status() -> None:
    """Print current target and XAI coverage status."""

    targets = build_targets()
    local = load_existing_local()
    coverage = coverage_table(targets, local)
    print(f"Target alarm episode representative rows: {len(targets):,}")
    if local.empty:
        print("Already explained target/method rows: 0")
    else:
        explained = local[["fold", "model", "row_id", "method"]].drop_duplicates()
        print(f"Already explained target/method rows: {len(explained):,}")
    print(
        f"Missing target/method rows: "
        f"{int(coverage['missing_cases'].sum()):,}"
    )
    print(coverage.to_string(index=False))


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--compute-missing", action="store_true")
    parser.add_argument("--build-tables", action="store_true")
    parser.add_argument("--limit-per-group", type=int, default=None)
    args = parser.parse_args()
    if args.status:
        print_status()
    if args.compute_missing:
        compute_missing_xai(limit_per_group=args.limit_per_group)
    if args.build_tables:
        build_evidence_and_agreement_tables()
        print(f"Wrote all-alarm XAI tables to {config.TABLES_DIR}")


if __name__ == "__main__":
    main()
