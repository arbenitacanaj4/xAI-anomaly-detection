"""Episode-level alarm triage for operator-oriented review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_incident_xai import config


def load_manifest(prefer_final: bool = True) -> list[dict[str, object]]:
    """Load final manifest when available, otherwise debug manifest."""

    path = (
        config.FINAL_MANIFEST_PATH
        if prefer_final and config.FINAL_MANIFEST_PATH.exists()
        else config.DEBUG_MANIFEST_PATH
    )
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_identity() -> pd.DataFrame:
    """Load row-level identity and traffic context columns."""

    columns = [
        "row_id",
        "stream_id",
        "source_file",
        "timestamp",
        "station_id",
        "row_in_split",
        "split",
        "speed_smoothed",
        "occ_smoothed",
        "is_incident",
    ]
    frame = pd.read_csv(config.PROCESSED_DIR / "training_dataset.csv", usecols=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def normalize_prediction_file(path: Path, identity: pd.DataFrame) -> pd.DataFrame:
    """Normalize old and new score-file schemas into one prediction schema."""

    frame = pd.read_csv(path)
    if "score" not in frame.columns and "anomaly_score" in frame.columns:
        frame["score"] = frame["anomaly_score"]
    if "model" not in frame.columns and "model_name" in frame.columns:
        frame["model"] = frame["model_name"]
    if "prediction" not in frame.columns and "alarm" in frame.columns:
        frame["prediction"] = frame["alarm"]
    if "true_label" not in frame.columns:
        frame["true_label"] = frame["is_incident"].astype(int)
    if "normalized_score" not in frame.columns:
        frame["normalized_score"] = (
            frame["score"] / (frame["score"] + frame["threshold"] + 1e-12)
        )
    if "threshold_margin" not in frame.columns:
        frame["threshold_margin"] = frame["score"] - frame["threshold"]
    if "case_type" not in frame.columns:
        frame["case_type"] = np.select(
            [
                (frame["prediction"] == 1) & (frame["true_label"] == 1),
                (frame["prediction"] == 1) & (frame["true_label"] == 0),
                (frame["prediction"] == 0) & (frame["true_label"] == 1),
            ],
            ["true_positive", "false_positive", "missed_incident"],
            default="true_negative",
        )
    identity_columns = ("source_file", "row_in_split", "speed_smoothed", "occ_smoothed")
    missing_identity = [column for column in identity_columns if column not in frame.columns]
    if missing_identity:
        frame = frame.merge(
            identity[["row_id", *missing_identity]],
            on="row_id",
            how="left",
        )
    for column in identity_columns:
        x_column = f"{column}_x"
        y_column = f"{column}_y"
        if column not in frame.columns and x_column in frame.columns:
            frame[column] = frame[x_column]
        if column not in frame.columns and y_column in frame.columns:
            frame[column] = frame[y_column]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def load_all_predictions() -> pd.DataFrame:
    """Load all fold/model prediction files from the active manifest."""

    identity = load_dataset_identity()
    frames = []
    for record in load_manifest():
        score_path = config.resolve_existing_path(str(record["score_path"]))
        if score_path.exists():
            frames.append(normalize_prediction_file(score_path, identity))
    if not frames:
        raise FileNotFoundError("No prediction score files found in manifest.")
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(config.FINAL_PREDICTIONS_DIR / "all_model_predictions.csv", index=False)
    return predictions


def confidence_level(mean_margin: float, peak_score: float) -> str:
    """Classify confidence without using labels."""

    if mean_margin > 0.05 and peak_score > 0.60:
        return "high"
    if mean_margin > 0.0:
        return "medium"
    return "borderline"


def temporal_pattern(duration: int, scores: pd.Series) -> str:
    """Classify alarm persistence/trend without labels."""

    if duration <= 2:
        return "short_spike"
    trend = float(scores.iloc[-1] - scores.iloc[0])
    if trend > 0.05:
        return "worsening"
    if trend < -0.05:
        return "recovering"
    if duration >= 6:
        return "persistent"
    return "borderline"


def label_alarm_type(
    pattern: str,
    confidence: str,
    agreement_count: int,
    explanation_level: str,
) -> str:
    """Assign an operational alarm type without using true labels."""

    if pattern == "short_spike":
        return "short spike false alarm candidate"
    if agreement_count <= 1:
        return "one-model-only alarm"
    if agreement_count >= 3:
        return "multi-model consensus alarm"
    if explanation_level == "conflicting":
        return "explanation-conflict alarm"
    if confidence == "high":
        return "high-confidence alarm"
    if pattern == "persistent":
        return "persistent congestion-like alarm"
    return "borderline alarm"


def reliability_level(score: float) -> str:
    """Convert support score to a compact operator label."""

    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def false_positive_risk_level(score: float) -> str:
    """Convert risk score to a compact operator label."""

    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def operator_priority(reliability: str, risk: str) -> str:
    """Assign review priority without claiming real-time truth."""

    if reliability == "high" and risk != "high":
        return "urgent review"
    if reliability == "medium":
        return "standard review"
    return "monitor"


def load_xai_support() -> pd.DataFrame:
    """Load row/model XAI support and agreement summaries when available."""

    support_columns = [
        "row_id",
        "model",
        "dominant_xai_features",
        "explanation_support_score",
        "explanation_agreement_level",
    ]
    if not config.XAI_LOCAL_EXPLANATIONS_PATH.exists():
        return pd.DataFrame(columns=support_columns)
    local = pd.read_csv(config.XAI_LOCAL_EXPLANATIONS_PATH)
    grouped = (
        local.groupby(["row_id", "model"])
        .agg(
            dominant_xai_features=(
                "feature",
                lambda values: ", ".join(list(values)[:3]),
            ),
            explanation_support_score=("abs_attribution", "sum"),
        )
        .reset_index()
    )
    if config.XAI_METHOD_AGREEMENT_PATH.exists():
        agreement = pd.read_csv(config.XAI_METHOD_AGREEMENT_PATH)
        key_columns = ["case_id", "model", "explanation_agreement_level"]
        local_keys = local[["case_id", "row_id", "model"]].drop_duplicates()
        agreement = agreement[key_columns].merge(
            local_keys,
            on=["case_id", "model"],
            how="left",
        )
        grouped = grouped.merge(
            agreement[["row_id", "model", "explanation_agreement_level"]],
            on=["row_id", "model"],
            how="left",
        )
    if "explanation_agreement_level" not in grouped.columns:
        grouped["explanation_agreement_level"] = "not_available"
    grouped["explanation_agreement_level"] = grouped[
        "explanation_agreement_level"
    ].fillna("not_available")
    return grouped[support_columns]


def incident_episode_summary(identity: pd.DataFrame) -> pd.DataFrame:
    """Summarize labeled incident periods for after-the-fact evaluation."""

    rows = []
    test = identity[identity["split"] == "test"].copy()
    for stream_id, group in test.groupby("stream_id"):
        incident = group[group["is_incident"] == 1]
        if incident.empty:
            continue
        rows.append(
            {
                "stream_id": int(stream_id),
                "source_file": str(group["source_file"].iloc[0]),
                "incident_start": incident["timestamp"].min(),
                "incident_end": incident["timestamp"].max(),
                "incident_duration_timestamps": int(len(incident)),
                "incident_duration_minutes": int(len(incident) * 5),
            }
        )
    return pd.DataFrame(rows)


def incident_level_detection_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute thesis-style incident-level detection metrics.

    Detection rate is counted per incident stream, not per timestamp. False alarm
    rate is the fraction of non-incident test timestamps that were alarmed.
    Mean time to detection uses the first alarm inside the labeled incident
    period and is reported only for detected incidents.
    """

    rows = []
    predictions = predictions.copy()
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    for (model, fold, stream_id), group in predictions.groupby(
        ["model", "fold", "stream_id"]
    ):
        ordered = group.sort_values("timestamp")
        incident = ordered[ordered["true_label"] == 1]
        normal = ordered[ordered["true_label"] == 0]
        if incident.empty:
            continue
        incident_alarms = incident[incident["prediction"] == 1]
        detected = not incident_alarms.empty
        first_alarm = pd.NaT
        detection_delay = np.nan
        if detected:
            first_alarm = incident_alarms["timestamp"].min()
            detection_delay = (
                first_alarm - incident["timestamp"].min()
            ).total_seconds() / 60.0
        false_alarm_timestamps = int(normal["prediction"].sum())
        normal_timestamps = int(len(normal))
        rows.append(
            {
                "model": model,
                "fold": int(fold),
                "stream_id": int(stream_id),
                "source_file": str(ordered["source_file"].iloc[0]),
                "incident_start": incident["timestamp"].min(),
                "incident_end": incident["timestamp"].max(),
                "incident_timestamps": int(len(incident)),
                "detected_incident": int(detected),
                "first_alarm_time": first_alarm,
                "detection_delay_minutes": detection_delay,
                "alarm_coverage_inside_incident": float(
                    incident["prediction"].mean()
                ),
                "missed_incident_timestamps": int(
                    len(incident) - incident["prediction"].sum()
                ),
                "false_alarm_timestamps": false_alarm_timestamps,
                "normal_timestamps": normal_timestamps,
                "false_alarm_rate": (
                    false_alarm_timestamps / normal_timestamps
                    if normal_timestamps
                    else np.nan
                ),
            }
        )
    incident_level = pd.DataFrame(rows)
    fold_metrics = (
        incident_level.groupby(["model", "fold"])
        .agg(
            incident_count=("stream_id", "count"),
            detected_incidents=("detected_incident", "sum"),
            false_alarm_timestamps=("false_alarm_timestamps", "sum"),
            normal_timestamps=("normal_timestamps", "sum"),
            mean_time_to_detection_minutes=(
                "detection_delay_minutes",
                "mean",
            ),
            median_time_to_detection_minutes=(
                "detection_delay_minutes",
                "median",
            ),
            mean_alarm_coverage_inside_incident=(
                "alarm_coverage_inside_incident",
                "mean",
            ),
        )
        .reset_index()
    )
    fold_metrics["missed_incidents"] = (
        fold_metrics["incident_count"] - fold_metrics["detected_incidents"]
    )
    fold_metrics["detection_rate"] = (
        fold_metrics["detected_incidents"] / fold_metrics["incident_count"]
    )
    fold_metrics["false_alarm_rate"] = (
        fold_metrics["false_alarm_timestamps"] / fold_metrics["normal_timestamps"]
    )
    model_metrics = (
        fold_metrics.groupby("model")
        .agg(
            detection_rate_mean=("detection_rate", "mean"),
            false_alarm_rate_mean=("false_alarm_rate", "mean"),
            mean_time_to_detection_minutes=(
                "mean_time_to_detection_minutes",
                "mean",
            ),
            median_time_to_detection_minutes=(
                "median_time_to_detection_minutes",
                "mean",
            ),
            mean_alarm_coverage_inside_incident=(
                "mean_alarm_coverage_inside_incident",
                "mean",
            ),
            incident_count_total=("incident_count", "sum"),
            detected_incidents_total=("detected_incidents", "sum"),
            missed_incidents_total=("missed_incidents", "sum"),
            false_alarm_timestamps_total=("false_alarm_timestamps", "sum"),
            normal_timestamps_total=("normal_timestamps", "sum"),
        )
        .reset_index()
    )
    model_metrics["metric_note"] = (
        "Thesis-style incident-level DR with timestamp-level FAR; "
        "MTTD is measured from incident start to first alarm inside the "
        "labeled incident period."
    )
    return incident_level, model_metrics


def build_alarm_episodes(predictions: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive positive predictions into alarm episodes."""

    xai_support = load_xai_support()
    rows = []
    agreement_lookup = (
        predictions[predictions["prediction"] == 1]
        .groupby("row_id")["model"]
        .nunique()
        .to_dict()
    )
    for (model, fold, stream_id), group in predictions.groupby(
        ["model", "fold", "stream_id"]
    ):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        positive = ordered["prediction"].astype(int)
        episode_group = (positive.diff().fillna(0) != 0).cumsum()
        ordered["episode_group"] = episode_group
        for _, episode in ordered[ordered["prediction"] == 1].groupby("episode_group"):
            peak_index = episode["normalized_score"].idxmax()
            peak = ordered.loc[peak_index]
            duration = int(len(episode))
            incident_overlap = int(episode["true_label"].sum())
            agreement_count = int(agreement_lookup.get(int(peak["row_id"]), 1))
            xai_row = xai_support[
                (xai_support["row_id"] == int(peak["row_id"]))
                & (xai_support["model"] == model)
            ]
            dominant_features = "not_available"
            support = 0.0
            agreement_level_value = "not_available"
            if not xai_row.empty:
                dominant_features = str(xai_row["dominant_xai_features"].iloc[0])
                support = float(xai_row["explanation_support_score"].iloc[0])
                agreement_level_value = str(
                    xai_row["explanation_agreement_level"].iloc[0]
                )
            pattern = temporal_pattern(duration, episode["normalized_score"])
            confidence = confidence_level(
                float(episode["threshold_margin"].mean()),
                float(episode["normalized_score"].max()),
            )
            reliability_score = min(
                1.0,
                0.35 * min(1.0, max(0.0, float(episode["threshold_margin"].mean())))
                + 0.30 * (agreement_count / 4.0)
                + 0.20 * min(1.0, duration / 6.0)
                + 0.15 * min(1.0, support),
            )
            risk_score = min(
                1.0,
                0.35 * (1.0 if confidence == "borderline" else 0.0)
                + 0.25 * (1.0 if agreement_count <= 1 else 0.0)
                + 0.20 * (1.0 if pattern == "short_spike" else 0.0)
                + 0.20 * (1.0 if agreement_level_value == "conflicting" else 0.0),
            )
            reliability = reliability_level(reliability_score)
            risk = false_positive_risk_level(risk_score)
            alarm_type = label_alarm_type(
                pattern,
                confidence,
                agreement_count,
                agreement_level_value,
            )
            rows.append(
                {
                    "model": model,
                    "fold": int(fold),
                    "stream_id": int(stream_id),
                    "source_file": str(episode["source_file"].iloc[0]),
                    "alarm_episode_id": len(rows) + 1,
                    "alarm_start": episode["timestamp"].min(),
                    "alarm_end": episode["timestamp"].max(),
                    "alarm_duration_timestamps": duration,
                    "alarm_duration_minutes": duration * 5,
                    "representative_timestamp": peak["timestamp"],
                    "representative_row_id": int(peak["row_id"]),
                    "peak_probability": float(episode["normalized_score"].max()),
                    "mean_probability": float(episode["normalized_score"].mean()),
                    "max_threshold_margin": float(episode["threshold_margin"].max()),
                    "mean_threshold_margin": float(episode["threshold_margin"].mean()),
                    "confidence_level": confidence,
                    "incident_overlap_timestamps": incident_overlap,
                    "incident_overlap_minutes": incident_overlap * 5,
                    "evaluation_episode_type": (
                        "true_incident_alarm"
                        if incident_overlap > 0
                        else "false_alarm_episode"
                    ),
                    "model_agreement_count": agreement_count,
                    "dominant_xai_features": dominant_features,
                    "explanation_support_score": support,
                    "explanation_agreement_level": agreement_level_value,
                    "probability_trend_before_alarm": pattern,
                    "speed_trend_around_alarm": float(
                        episode["speed_smoothed"].iloc[-1]
                        - episode["speed_smoothed"].iloc[0]
                    ),
                    "occupancy_trend_around_alarm": float(
                        episode["occ_smoothed"].iloc[-1]
                        - episode["occ_smoothed"].iloc[0]
                    ),
                    "temporal_pattern": pattern,
                    "alarm_type": alarm_type,
                    "reliability_level": reliability,
                    "false_positive_risk_level": risk,
                    "operator_priority": operator_priority(reliability, risk),
                    "operator_note": (
                        "Use as supporting information for operator judgment; "
                        "XAI describes model behavior, not physical causality."
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_alarm_cards(episodes: pd.DataFrame, limit: int = 8) -> None:
    """Write compact example operator cards."""

    config.ALARM_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    selected = episodes.sort_values(
        ["operator_priority", "peak_probability"],
        ascending=[True, False],
    ).head(limit)
    cards = []
    for _, row in selected.iterrows():
        cards.append(
            "\n".join(
                [
                    f"Alarm episode {row['alarm_episode_id']} ({row['model']})",
                    f"Stream: {row['stream_id']} | Time: {row['alarm_start']} to {row['alarm_end']}",
                    f"Priority: {row['operator_priority']} | Reliability: {row['reliability_level']} | FP risk: {row['false_positive_risk_level']}",
                    f"Evidence: {row['dominant_xai_features']}",
                    f"Pattern: {row['temporal_pattern']} | Model agreement: {row['model_agreement_count']}/4",
                    f"Note: {row['operator_note']}",
                ]
            )
        )
    config.ALARM_CARD_EXAMPLES_PATH.write_text(
        "\n\n---\n\n".join(cards),
        encoding="utf-8",
    )


def build_triage_outputs() -> None:
    """Create incident summaries, alarm episode triage, and cards."""

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FINAL_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    identity = load_dataset_identity()
    predictions = load_all_predictions()
    incidents = incident_episode_summary(identity)
    incident_level, thesis_metrics = incident_level_detection_tables(predictions)
    episodes = build_alarm_episodes(predictions)
    incidents.to_csv(config.INCIDENT_EPISODE_SUMMARY_PATH, index=False)
    incident_level.to_csv(config.INCIDENT_LEVEL_DETECTION_PATH, index=False)
    thesis_metrics.to_csv(config.THESIS_STYLE_MODEL_METRICS_PATH, index=False)
    episodes.to_csv(config.ALARM_EPISODE_TRIAGE_PATH, index=False)
    episodes[
        episodes["evaluation_episode_type"] == "false_alarm_episode"
    ].to_csv(config.FALSE_ALARM_TRIAGE_PATH, index=False)
    write_alarm_cards(episodes)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build_triage_outputs()
    print(f"Wrote alarm triage outputs to {config.TABLES_DIR}.")


if __name__ == "__main__":
    main()
