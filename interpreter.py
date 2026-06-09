"""Operator-oriented diagnostic messages from numerical SHAP arrays."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AttributionSummary:
    """Compact signal summary used by the operator logic layer."""

    speed_total: float
    occupancy_total: float
    speed_peak_ratio: float
    occupancy_peak_ratio: float
    speed_occupancy_ratio: float
    peak_alignment: float
    speed_graduality: float
    occupancy_graduality: float


def safe_sum(values: np.ndarray) -> float:
    """Return a stable absolute attribution total."""

    return float(np.sum(np.abs(values)) + 1e-12)


def normalize_series(values: np.ndarray) -> np.ndarray:
    """Normalize a non-negative attribution series."""

    absolute = np.abs(values).astype(float)
    total = np.sum(absolute)
    if total <= 1e-12:
        return np.zeros_like(absolute)
    return absolute / total


def infer_feature_groups(
    shap_values: np.ndarray,
    feature_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract speed and occupancy attribution series."""

    values = np.asarray(shap_values)
    if values.ndim == 3:
        speed_values = values[:, :, 0].mean(axis=0)
        occupancy_values = values[:, :, 1].mean(axis=0)
        return speed_values, occupancy_values

    names = [str(name).lower() for name in feature_names.tolist()]
    speed_indices = [index for index, name in enumerate(names) if "speed" in name]
    occupancy_indices = [
        index
        for index, name in enumerate(names)
        if "occ" in name or "occupancy" in name
    ]
    if not speed_indices or not occupancy_indices:
        raise ValueError("Could not identify speed and occupancy feature groups.")
    speed_values = values[:, speed_indices].mean(axis=0)
    occupancy_values = values[:, occupancy_indices].mean(axis=0)
    return speed_values, occupancy_values


def summarize_attributions(
    shap_values: np.ndarray,
    feature_names: np.ndarray,
) -> AttributionSummary:
    """Calculate attribution ratios for diagnostic rules."""

    speed_values, occupancy_values = infer_feature_groups(shap_values, feature_names)
    speed_norm = normalize_series(speed_values)
    occupancy_norm = normalize_series(occupancy_values)
    speed_total = safe_sum(speed_values)
    occupancy_total = safe_sum(occupancy_values)
    speed_peak_index = int(np.argmax(speed_norm)) if len(speed_norm) else 0
    occupancy_peak_index = int(np.argmax(occupancy_norm)) if len(occupancy_norm) else 0
    speed_peak_ratio = float(np.max(speed_norm)) if len(speed_norm) else 0.0
    occupancy_peak_ratio = float(np.max(occupancy_norm)) if len(occupancy_norm) else 0.0
    peak_distance = abs(speed_peak_index - occupancy_peak_index)
    peak_alignment = 1.0 / (1.0 + float(peak_distance))
    speed_gradient = np.abs(np.diff(speed_norm)).sum() if len(speed_norm) > 1 else 0.0
    occupancy_gradient = (
        np.abs(np.diff(occupancy_norm)).sum() if len(occupancy_norm) > 1 else 0.0
    )
    return AttributionSummary(
        speed_total=speed_total,
        occupancy_total=occupancy_total,
        speed_peak_ratio=speed_peak_ratio,
        occupancy_peak_ratio=occupancy_peak_ratio,
        speed_occupancy_ratio=speed_total / occupancy_total,
        peak_alignment=peak_alignment,
        speed_graduality=1.0 / (1.0 + float(speed_gradient)),
        occupancy_graduality=1.0 / (1.0 + float(occupancy_gradient)),
    )


def diagnose(summary: AttributionSummary) -> str:
    """Return a plain-text diagnostic message for a transport operator."""

    if (
        summary.speed_occupancy_ratio >= 4.0
        and summary.speed_peak_ratio >= 0.45
        and summary.occupancy_peak_ratio <= 0.25
    ):
        return (
            "Sensor malfunction flag: the alarm is dominated by an extreme, "
            "localized speed attribution while occupancy evidence stays flat. "
            "Check detector health and upstream sensor telemetry before treating "
            "this as a confirmed road incident."
        )
    if (
        summary.speed_graduality >= 0.55
        and summary.speed_peak_ratio <= 0.30
        and summary.occupancy_peak_ratio <= 0.35
    ):
        return (
            "Routine congestion flag: speed evidence accumulates gradually and "
            "does not pair with a sharp occupancy spike. This looks closer to "
            "baseline congestion drift than a sudden incident signature."
        )
    if (
        summary.speed_peak_ratio >= 0.35
        and summary.occupancy_peak_ratio >= 0.35
        and summary.peak_alignment >= 0.50
    ):
        return (
            "True incident profile: a sudden speed-drop attribution aligns with "
            "a localized occupancy spike. Prioritize operator review and nearby "
            "camera or patrol confirmation."
        )
    return (
        "Mixed evidence: attribution ratios do not strongly match malfunction, "
        "routine congestion, or a clean incident profile. Review adjacent time "
        "steps and compare model agreement before escalation."
    )


def interpret_file(path: Path) -> str:
    """Load an explanation artifact and produce an operator diagnostic."""

    artifact = np.load(path, allow_pickle=True)
    shap_values = artifact["shap_values"]
    feature_names = artifact["feature_names"]
    summary = summarize_attributions(shap_values, feature_names)
    lines = [
        diagnose(summary),
        "",
        "Attribution ratios:",
        f"- speed_total={summary.speed_total:.6f}",
        f"- occupancy_total={summary.occupancy_total:.6f}",
        f"- speed_occupancy_ratio={summary.speed_occupancy_ratio:.3f}",
        f"- speed_peak_ratio={summary.speed_peak_ratio:.3f}",
        f"- occupancy_peak_ratio={summary.occupancy_peak_ratio:.3f}",
        f"- peak_alignment={summary.peak_alignment:.3f}",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(interpret_file(args.artifact))


if __name__ == "__main__":
    main()
