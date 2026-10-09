"""Report-ready figure generation for the final NITA XAI project outputs."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd

from traffic_incident_xai import config


MODEL_ORDER = ["random_forest", "xgboost", "mlp", "lstm"]
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "lstm": "LSTM",
}
METHOD_LABELS = {
    "treeshap": "TreeSHAP",
    "deepshap": "DeepSHAP-style",
    "sequence_deepshap": "Sequence DeepSHAP",
    "lime_style_local_surrogate": "LIME-style local surrogate",
    "lime": "LIME-style local surrogate",
}
MODEL_COLORS = {
    "random_forest": "#2C7FB8",
    "xgboost": "#41AB5D",
    "mlp": "#F16913",
    "lstm": "#756BB1",
}
REPORT_FIG_DIR = config.RESULTS_DIR / "figures" / "report"
MANIFEST_PATH = REPORT_FIG_DIR / "report_figure_manifest.csv"


@dataclass
class ReportFigure:
    """Metadata for one generated report figure."""

    filename: str
    source_tables: str
    notebook_section: str
    suggested_report_section: str
    short_caption: str
    interpretation_note: str


@dataclass
class ReportContext:
    """Loaded final project outputs plus generated figure metadata."""

    training: pd.DataFrame
    final_metrics: pd.DataFrame
    final_comparison: pd.DataFrame
    final_thresholds: pd.DataFrame
    thesis_metrics: pd.DataFrame
    incident_detection: pd.DataFrame
    predictions: pd.DataFrame
    alarm_episodes: pd.DataFrame
    false_alarm_triage: pd.DataFrame
    incident_summary: pd.DataFrame
    xai_targets: pd.DataFrame
    xai_coverage: pd.DataFrame
    xai_evidence: pd.DataFrame
    xai_method_agreement: pd.DataFrame
    xai_cross_model: pd.DataFrame
    xai_local_row_count: int
    manifest_rows: list[ReportFigure] = field(default_factory=list)


def configure_style() -> None:
    """Apply a clean academic Matplotlib style."""

    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def model_label(value: str) -> str:
    return MODEL_LABELS.get(str(value), str(value).replace("_", " ").title())


def method_label(value: str) -> str:
    return METHOD_LABELS.get(str(value), str(value).replace("_", " ").title())


def feature_label(feature: str) -> str:
    text = str(feature)
    if text == "speed_smoothed":
        return "Speed"
    if text == "occ_smoothed":
        return "Occupancy"
    if "speed_smoothed_t_minus_" in text:
        lag = int(text.rsplit("_", 1)[-1])
        return f"Speed, {lag * 5} min before"
    if "occ_smoothed_t_minus_" in text:
        lag = int(text.rsplit("_", 1)[-1])
        return f"Occupancy, {lag * 5} min before"
    return text.replace("_", " ").title()


def clean_level(value: str) -> str:
    return str(value).replace("_", " ").title()


def wrap_text(value: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width))


def load_context() -> ReportContext:
    """Load final outputs used by the report notebook."""

    configure_style()
    tables = config.TABLES_DIR
    local_path = tables / "xai_local_explanations_all_alarm_episodes.csv"
    return ReportContext(
        training=pd.read_csv(
            config.PROCESSED_DIR / "training_dataset.csv",
            parse_dates=["timestamp"],
        ),
        final_metrics=pd.read_csv(tables / "final_model_metrics.csv"),
        final_comparison=pd.read_csv(tables / "final_model_comparison.csv"),
        final_thresholds=pd.read_csv(tables / "final_thresholds.csv"),
        thesis_metrics=pd.read_csv(tables / "thesis_style_model_metrics.csv"),
        incident_detection=pd.read_csv(
            tables / "incident_level_detection.csv",
            parse_dates=["incident_start", "incident_end", "first_alarm_time"],
        ),
        predictions=pd.read_csv(
            config.FINAL_PREDICTIONS_DIR / "all_model_predictions.csv",
            parse_dates=["timestamp"],
        ),
        alarm_episodes=pd.read_csv(
            config.ALARM_EPISODE_TRIAGE_PATH,
            parse_dates=["alarm_start", "alarm_end", "representative_timestamp"],
        ),
        false_alarm_triage=pd.read_csv(
            config.FALSE_ALARM_TRIAGE_PATH,
            parse_dates=["alarm_start", "alarm_end", "representative_timestamp"],
        ),
        incident_summary=pd.read_csv(
            config.INCIDENT_EPISODE_SUMMARY_PATH,
            parse_dates=["incident_start", "incident_end"],
        ),
        xai_targets=pd.read_csv(tables / "xai_alarm_episode_targets.csv"),
        xai_coverage=pd.read_csv(tables / "xai_alarm_episode_coverage.csv"),
        xai_evidence=pd.read_csv(tables / "xai_alarm_evidence_metrics.csv"),
        xai_method_agreement=pd.read_csv(
            tables / "xai_method_agreement_all_alarm_episodes.csv"
        ),
        xai_cross_model=pd.read_csv(
            tables / "xai_cross_model_agreement_all_alarm_episodes.csv"
        ),
        xai_local_row_count=len(pd.read_csv(local_path, usecols=["case_id"])),
    )


def save_figure(
    ctx: ReportContext,
    fig: plt.Figure,
    filename: str,
    source_tables: str,
    notebook_section: str,
    suggested_report_section: str,
    short_caption: str,
    interpretation_note: str,
) -> None:
    """Save a figure as a high-resolution PNG and record its metadata."""

    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = REPORT_FIG_DIR / filename
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    ctx.manifest_rows.append(
        ReportFigure(
            filename=filename,
            source_tables=source_tables,
            notebook_section=notebook_section,
            suggested_report_section=suggested_report_section,
            short_caption=short_caption,
            interpretation_note=interpretation_note,
        )
    )


def project_pipeline_overview(ctx: ReportContext) -> plt.Figure:
    steps = [
        ("1", "Incident files", "Raw + smoothed"),
        ("2", "Dataset", "175 streams"),
        ("3", "5-fold split", "Stream-level CV"),
        ("4", "Train models", "RF, XGB, MLP, LSTM"),
        ("5", "Score alarms", "Thresholded anomaly scores"),
        ("6", "Group episodes", "Consecutive alarms"),
        ("7", "Explain alarms", "SHAP, LIME, sequence XAI"),
        ("8", "Triage output", "Operator evidence card"),
    ]
    box_colors = [
        "#EAF4FF",
        "#EAF4FF",
        "#ECF8F0",
        "#ECF8F0",
        "#ECF8F0",
        "#FFF4DD",
        "#F3ECFF",
        "#F3ECFF",
    ]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    positions = [
        (0.15, 0.68),
        (0.38, 0.68),
        (0.61, 0.68),
        (0.84, 0.68),
        (0.84, 0.30),
        (0.61, 0.30),
        (0.38, 0.30),
        (0.15, 0.30),
    ]
    phase_labels = [
        (0.265, 0.875, 0.34, "Data preparation", "#DCEEFF"),
        (0.725, 0.875, 0.34, "Modeling", "#E9F7EF"),
        (0.725, 0.495, 0.34, "Alarm layer", "#FFF1D6"),
        (0.265, 0.495, 0.34, "XAI triage", "#F0E8FF"),
    ]
    box_width = 0.17
    box_height = 0.19

    for center_x, center_y, width, label, color in phase_labels:
        ribbon = patches.FancyBboxPatch(
            (center_x - width / 2, center_y - 0.035),
            width,
            0.07,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0,
            facecolor=color,
            alpha=0.98,
        )
        ax.add_patch(ribbon)
        ax.text(
            center_x,
            center_y,
            label,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="#243447",
        )

    for index, ((x_pos, y_pos), (number, title, detail)) in enumerate(
        zip(positions, steps)
    ):
        shadow = patches.FancyBboxPatch(
            (x_pos - box_width / 2 + 0.008, y_pos - box_height / 2 - 0.01),
            box_width,
            box_height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0,
            facecolor="#CAD4DF",
            alpha=0.35,
        )
        ax.add_patch(shadow)
        box = patches.FancyBboxPatch(
            (x_pos - box_width / 2, y_pos - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.8,
            edgecolor="#2F4054",
            facecolor=box_colors[index],
        )
        ax.add_patch(box)
        badge = patches.Circle(
            (x_pos, y_pos + box_height / 2 - 0.012),
            0.025,
            facecolor="#2F4054",
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(badge)
        ax.text(
            x_pos,
            y_pos + box_height / 2 - 0.012,
            number,
            ha="center",
            va="center",
            fontsize=13,
            color="white",
            fontweight="bold",
        )
        ax.text(
            x_pos,
            y_pos + 0.02,
            title,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#111827",
        )
        ax.text(
            x_pos,
            y_pos - 0.055,
            detail,
            ha="center",
            va="center",
            fontsize=12.5,
            color="#374151",
            linespacing=1.15,
        )

    arrow_pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
    for left_index, right_index in arrow_pairs:
        x1, y1 = positions[left_index]
        x2, y2 = positions[right_index]
        if x2 > x1:
            start = (x1 + box_width / 2 + 0.02, y1)
            end = (x2 - box_width / 2 - 0.02, y2)
        else:
            start = (x1 - box_width / 2 - 0.02, y1)
            end = (x2 + box_width / 2 + 0.02, y2)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 2.4,
                "color": "#2F4054",
                "mutation_scale": 18,
            },
        )
    ax.annotate(
        "",
        xy=(positions[4][0], positions[4][1] + box_height / 2 + 0.035),
        xytext=(positions[3][0], positions[3][1] - box_height / 2 - 0.035),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 2.4,
            "color": "#2F4054",
            "mutation_scale": 18,
        },
    )
    ax.set_title(
        "End-to-end XAI alarm triage workflow",
        fontsize=25,
        fontweight="bold",
        color="#111827",
        pad=18,
    )
    ax.text(
        0.5,
        0.075,
        "The project turns traffic streams into model alarms, groups alarms into episodes, and explains each episode for operator review.",
        ha="center",
        va="center",
        fontsize=15,
        color="#374151",
        wrap=True,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.93)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.07)
    save_figure(
        ctx,
        fig,
        "project_pipeline_overview_presentation.png",
        "training_dataset.csv; final_model_metrics.csv; all_model_predictions.csv; alarm_episode_triage.csv; xai_alarm_evidence_metrics.csv",
        "Project/data protocol figures",
        "Method overview",
        "Presentation-ready workflow from prepared incident-window files to final operator-facing triage output.",
        "The project is a complete alarm-triage workflow, not only a model-performance comparison.",
    )
    save_figure(
        ctx,
        fig,
        "project_pipeline_overview.png",
        "training_dataset.csv; final_model_metrics.csv; all_model_predictions.csv; alarm_episode_triage.csv; xai_alarm_evidence_metrics.csv",
        "Project/data protocol figures",
        "Method overview",
        "Project workflow from prepared incident-window files to final operator-facing triage output.",
        "The project is a complete alarm-triage workflow, not only a model-performance comparison.",
    )
    return fig


def incident_window_protocol(ctx: ReportContext) -> plt.Figure:
    stream_id = int(ctx.incident_summary.iloc[0]["stream_id"])
    test_rows = ctx.training[
        (ctx.training["stream_id"] == stream_id) & (ctx.training["split"] == "test")
    ]
    incident_rows = test_rows[test_rows["is_incident"] == 1]
    incident_start = 240 + incident_rows["row_in_split"].min() * 5 / 60
    incident_duration = len(incident_rows) * 5 / 60

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.broken_barh([(0, 240)], (0.35, 0.3), facecolors="#D9EAF7")
    ax.broken_barh([(240, 12)], (0.35, 0.3), facecolors="#FEE8C8")
    ax.broken_barh([(incident_start, incident_duration)], (0.35, 0.3), facecolors="#D7301F")
    ax.text(120, 0.72, "240-hour incident-free baseline / training window", ha="center")
    ax.text(246, 0.72, "12-hour test window", ha="center")
    ax.text(
        incident_start + incident_duration / 2,
        0.22,
        "labeled\nincident",
        ha="center",
        va="top",
        fontsize=9,
        color="#7F0000",
    )
    ax.set_xlim(-5, 255)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("Hours from start of stream")
    ax.set_title(f"Incident-window protocol for one stream, 5-minute sampling (stream {stream_id})")
    save_figure(
        ctx,
        fig,
        "incident_window_protocol.png",
        "training_dataset.csv; incident_episode_summary.csv",
        "Project/data protocol figures",
        "Data protocol",
        "One stream contains a 240-hour baseline window followed by a 12-hour test window with a labeled incident period.",
        "The split preserves a normal baseline before evaluating alarms inside the held-out incident window.",
    )
    return fig


def stream_level_cv_diagram(ctx: ReportContext) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    for fold in range(5):
        for block in range(5):
            is_test = block == fold
            rect = patches.Rectangle(
                (block, 4 - fold),
                0.9,
                0.72,
                facecolor="#F16913" if is_test else "#D9EAF7",
                edgecolor="#7F2704" if is_test else "#2C7FB8",
                linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(
                block + 0.45,
                4 - fold + 0.36,
                "Held-out\n35 streams" if is_test else "Train\n140 streams",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_xticks(np.arange(5) + 0.45)
    ax.set_xticklabels([f"Fold group {i}" for i in range(1, 6)])
    ax.set_yticks(np.arange(5) + 0.36)
    ax.set_yticklabels([f"CV run {i}" for i in range(5, 0, -1)])
    ax.set_xlabel("Stream groups kept intact")
    ax.set_ylabel("Cross-validation run")
    ax.set_title("Five-fold stream-level cross-validation: 175 streams = 5 groups x 35 streams")
    save_figure(
        ctx,
        fig,
        "stream_level_cv_diagram.png",
        "training_dataset.csv; final_model_metrics.csv",
        "Project/data protocol figures",
        "Validation protocol",
        "Five-fold stream-level cross-validation keeps complete streams together and holds out each 35-stream group once.",
        "The fold split tests generalization to unseen incident streams while avoiding mixing rows from the same stream across train and test.",
    )
    return fig


def dataset_label_window_summary(ctx: ReportContext) -> plt.Figure:
    split_counts = ctx.training.groupby("split").size().reindex(["train", "test"])
    test_labels = (
        ctx.training[ctx.training["split"] == "test"]
        .groupby("is_incident")
        .size()
        .rename(index={0: "Non-incident rows", 1: "Incident rows"})
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(["Train\n240h", "Test\n12h"], split_counts.values, color=["#2C7FB8", "#F16913"])
    axes[0].set_title("Rows by window split")
    axes[0].set_ylabel("5-minute rows")
    axes[1].bar(test_labels.index.astype(str), test_labels.values, color=["#BDBDBD", "#D7301F"])
    axes[1].set_title("Test rows by label")
    axes[1].set_ylabel("5-minute rows")
    axes[1].tick_params(axis="x", rotation=15)
    axes[2].hist(ctx.incident_summary["incident_duration_minutes"], bins=12, color="#756BB1", edgecolor="white")
    axes[2].set_title("Labeled incident duration")
    axes[2].set_xlabel("Minutes")
    axes[2].set_ylabel("Streams")
    fig.suptitle("Dataset window and label summary", y=1.03)
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "dataset_label_window_summary.png",
        "training_dataset.csv; incident_episode_summary.csv",
        "Project/data protocol figures",
        "Dataset summary",
        "Training/test row counts, test-window label balance, and labeled incident duration distribution.",
        "The data are highly imbalanced at timestamp level, so recall, incident detection, and alarm episodes are more informative than accuracy alone.",
    )
    return fig


def model_performance_figures(ctx: ReportContext) -> list[plt.Figure]:
    thesis = ctx.thesis_metrics.set_index("model").loc[MODEL_ORDER].reset_index()
    thesis["model_label"] = thesis["model"].map(model_label)
    colors = [MODEL_COLORS[m] for m in thesis["model"]]
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    bars0 = axes[0].bar(thesis["model_label"], thesis["detection_rate_mean"], color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Incident detection rate")
    axes[0].set_ylabel("Detected incidents / incidents")
    bars1 = axes[1].bar(thesis["model_label"], thesis["false_alarm_rate_mean"], color=colors)
    axes[1].set_title("False alarm rate")
    axes[1].set_ylabel("False alarms / normal timestamps")
    bars2 = axes[2].bar(thesis["model_label"], thesis["mean_time_to_detection_minutes"], color=colors)
    axes[2].set_title("Mean time to detection")
    axes[2].set_ylabel("Minutes")
    for bars, axis, fmt, pad in [
        (bars0, axes[0], "{:.1%}", 0.02),
        (bars1, axes[1], "{:.4f}", 0.0015),
        (bars2, axes[2], "{:.2f}", 0.03),
    ]:
        for bar in bars:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + pad,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    axes[1].set_ylim(0, thesis["false_alarm_rate_mean"].max() * 1.22)
    axes[2].set_ylim(0, thesis["mean_time_to_detection_minutes"].max() * 1.20)
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig1.suptitle("Incident-level performance comparison", y=1.03)
    fig1.tight_layout()
    save_figure(
        ctx,
        fig1,
        "incident_level_model_comparison.png",
        "thesis_style_model_metrics.csv",
        "Model evaluation figures",
        "Model evaluation",
        "Incident-level detection rate, false alarm rate, and mean time to detection for the four model families.",
        "This view matches the incident-window task: the key question is whether an incident receives an alarm.",
    )

    comparison = ctx.final_comparison.set_index("model_name").loc[MODEL_ORDER].reset_index()
    comparison["model_label"] = comparison["model_name"].map(model_label)
    x_pos = np.arange(len(comparison))
    fig2, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.25
    ax.bar(x_pos - width, comparison["precision_mean"], width, label="Precision", color="#9ECAE1")
    ax.bar(x_pos, comparison["recall_mean"], width, label="Recall", color="#41AB5D")
    ax.bar(x_pos + width, comparison["f1_mean"], width, label="F1", color="#F16913")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison["model_label"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Timestamp-level model comparison")
    ax.legend(ncol=3, frameon=False)
    fig2.tight_layout()
    save_figure(
        ctx,
        fig2,
        "timestamp_level_model_comparison.png",
        "final_model_comparison.csv",
        "Model evaluation figures",
        "Model evaluation",
        "Timestamp-level precision, recall, and F1 for final model outputs.",
        "Timestamp metrics show row-level behavior, while incident-level metrics show whether whole incidents were detected.",
    )
    return [fig1, fig2]


def fold_performance_stability(ctx: ReportContext) -> plt.Figure:
    stability = ctx.final_metrics.pivot_table(
        index="model_name", columns="fold", values="recall", aggfunc="mean"
    ).reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    image = ax.imshow(stability.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(stability.shape[1]))
    ax.set_xticklabels([f"Fold {fold}" for fold in stability.columns])
    ax.set_yticks(np.arange(stability.shape[0]))
    ax.set_yticklabels([model_label(model) for model in stability.index])
    for i in range(stability.shape[0]):
        for j in range(stability.shape[1]):
            ax.text(j, i, f"{stability.values[i, j]:.2f}", ha="center", va="center", color="#08306B")
    fig.colorbar(image, ax=ax, label="Recall")
    ax.set_title("Per-fold timestamp recall stability")
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "fold_performance_stability.png",
        "final_model_metrics.csv",
        "Model evaluation figures",
        "Model evaluation",
        "Fold-level timestamp recall heatmap for final model outputs.",
        "The heatmap checks whether a model performs consistently across held-out stream groups.",
    )
    return fig


def threshold_score_overview(ctx: ReportContext) -> plt.Figure:
    thresholds = ctx.final_thresholds.set_index("model_name").loc[MODEL_ORDER].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for model in MODEL_ORDER:
        values = thresholds[thresholds["model_name"] == model]["threshold"]
        axes[0].plot([model_label(model)] * len(values), values, "o", color=MODEL_COLORS[model])
    axes[0].set_title("Fold thresholds by model")
    axes[0].set_ylabel("Anomaly threshold")
    axes[0].tick_params(axis="x", rotation=20)
    box_data = [
        ctx.predictions[ctx.predictions["model"] == model]["threshold_margin"].clip(-0.25, 0.25)
        for model in MODEL_ORDER
    ]
    axes[1].boxplot(box_data, labels=[model_label(model) for model in MODEL_ORDER], showfliers=False)
    axes[1].axhline(0, color="#D7301F", linestyle="--", linewidth=1.2, label="Decision threshold")
    axes[1].set_title("Score margin around threshold")
    axes[1].set_ylabel("Score margin above threshold")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "threshold_score_overview.png",
        "final_thresholds.csv; all_model_predictions.csv",
        "Model evaluation figures",
        "Thresholds and scores",
        "Fold thresholds and score-margin distributions relative to each model's decision threshold.",
        "Score margin is useful for triage because borderline alarms differ from alarms far above threshold.",
    )
    return fig


def example_alarm_timeline(ctx: ReportContext) -> plt.Figure:
    candidate = ctx.alarm_episodes.sort_values(
        ["model_agreement_count", "peak_probability"], ascending=[False, False]
    ).iloc[0]
    stream_id = int(candidate["stream_id"])
    stream_data = ctx.training[(ctx.training["stream_id"] == stream_id) & (ctx.training["split"] == "test")]
    stream_preds = ctx.predictions[ctx.predictions["stream_id"] == stream_id]
    incidents = ctx.incident_summary[ctx.incident_summary["stream_id"] == stream_id]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(stream_data["timestamp"], stream_data["speed_smoothed"], color="#2C7FB8", linewidth=1.8)
    axes[1].plot(stream_data["timestamp"], stream_data["occ_smoothed"], color="#D7301F", linewidth=1.8)
    axes[0].set_ylabel("Speed")
    axes[1].set_ylabel("Occupancy")
    axes[1].set_xlabel("Test-window timestamp")
    for _, incident in incidents.iterrows():
        for ax in axes:
            ax.axvspan(incident["incident_start"], incident["incident_end"], color="#FEE0D2", alpha=0.75)
    for model in MODEL_ORDER:
        times = stream_preds[(stream_preds["model"] == model) & (stream_preds["prediction"] == 1)]["timestamp"]
        if len(times):
            axes[0].scatter(
                times,
                np.full(len(times), axes[0].get_ylim()[0]),
                s=24,
                marker="|",
                color=MODEL_COLORS[model],
                label=f"{model_label(model)} alarm",
            )
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[0].set_title(f"Example alarm timeline, stream {stream_id}")
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "example_alarm_timeline.png",
        "training_dataset.csv; all_model_predictions.csv; incident_episode_summary.csv",
        "Alarm episode figures",
        "Alarm examples",
        "Example test-window timeline showing speed, occupancy, labeled incident period, and model alarm timestamps.",
        "The timeline links model alarms to the smoothed speed and occupancy inputs.",
    )
    return fig


def timestamp_to_alarm_episode_grouping(ctx: ReportContext) -> plt.Figure:
    episode = ctx.alarm_episodes.sort_values(
        ["alarm_duration_minutes", "model_agreement_count"], ascending=[False, False]
    ).iloc[0]
    model = str(episode["model"])
    stream_id = int(episode["stream_id"])
    episode_preds = ctx.predictions[
        (ctx.predictions["model"] == model) & (ctx.predictions["stream_id"] == stream_id)
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.plot(
        episode_preds["timestamp"],
        episode_preds["prediction"],
        drawstyle="steps-post",
        color=MODEL_COLORS.get(model, "#333333"),
        linewidth=2,
    )
    ax.axvspan(episode["alarm_start"], episode["alarm_end"], color="#FEE8C8", alpha=0.85, label="Grouped alarm episode")
    ax.axvline(episode["representative_timestamp"], color="#D7301F", linewidth=2, label="Representative row")
    ax.set_ylim(-0.1, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No alarm", "Alarm"])
    ax.set_xlabel("Test-window timestamp")
    ax.set_title(f"Timestamp predictions grouped into one alarm episode ({model_label(model)}, stream {stream_id})")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "timestamp_to_alarm_episode_grouping.png",
        "all_model_predictions.csv; alarm_episode_triage.csv",
        "Alarm episode figures",
        "Alarm episodes",
        "Consecutive alarm-positive timestamps are grouped into one operational alarm episode with a representative peak row.",
        "Episode grouping prevents one operator card per positive 5-minute timestamp.",
    )
    return fig


def alarm_episode_summary(ctx: ReportContext) -> plt.Figure:
    episode_counts = ctx.alarm_episodes.groupby("model").size().reindex(MODEL_ORDER)
    agreement_counts = ctx.alarm_episodes["model_agreement_count"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].bar([model_label(model) for model in MODEL_ORDER], episode_counts.values, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    axes[0].set_title("Alarm episodes by model")
    axes[0].set_ylabel("Episode count")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].hist(ctx.alarm_episodes["alarm_duration_minutes"].clip(upper=120), bins=20, color="#74A9CF", edgecolor="white")
    axes[1].set_title("Alarm episode duration")
    axes[1].set_xlabel("Duration, minutes (clipped at 120)")
    axes[1].set_ylabel("Episodes")
    axes[2].bar(agreement_counts.index.astype(str), agreement_counts.values, color="#756BB1")
    axes[2].set_title("Model agreement at representative row")
    axes[2].set_xlabel("Number of models alarming")
    axes[2].set_ylabel("Episodes")
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "alarm_episode_summary.png",
        "alarm_episode_triage.csv",
        "Alarm episode figures",
        "Alarm episodes",
        "Alarm episode counts, duration distribution, and model agreement counts.",
        "This describes operator workload and how often alarms are supported by multiple model families.",
    )
    return fig


def xai_coverage_overview(ctx: ReportContext) -> plt.Figure:
    values = pd.Series(
        {
            "Alarm episode\nrepresentatives": len(ctx.xai_targets),
            "Explained model-method\ncombinations": len(ctx.xai_evidence),
            "Missing XAI\ncombinations": int(ctx.xai_coverage["missing_cases"].sum()),
            "Local attribution\nrows": ctx.xai_local_row_count,
            "Method-agreement\nrows": len(ctx.xai_method_agreement),
            "Cross-model agreement\nrows": len(ctx.xai_cross_model),
        }
    )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(values.index, values.values, color=["#2C7FB8", "#41AB5D", "#BDBDBD", "#756BB1", "#F16913", "#636363"])
    ax.set_title("All-episode XAI coverage overview")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for index, value in enumerate(values.values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "xai_coverage_overview.png",
        "xai_alarm_episode_targets.csv; xai_alarm_episode_coverage.csv; xai_alarm_evidence_metrics.csv; xai_local_explanations_all_alarm_episodes.csv; xai_method_agreement_all_alarm_episodes.csv; xai_cross_model_agreement_all_alarm_episodes.csv",
        "XAI evidence figures",
        "XAI coverage",
        "Coverage counts for all alarm episode representative rows and explanation agreement tables.",
        "The XAI results describe the whole alarm set, not only selected examples.",
    )
    return fig


def temporal_attribution_heatmap(ctx: ReportContext) -> plt.Figure:
    temporal = ctx.xai_evidence.groupby(["model", "method"])[
        ["recent_0_15_min_share", "mid_15_30_min_share", "older_30_60_min_share"]
    ].mean().reset_index()
    temporal["sort_model"] = temporal["model"].map({model: i for i, model in enumerate(MODEL_ORDER)})
    temporal = temporal.sort_values(["sort_model", "method"])
    temporal["label"] = temporal.apply(lambda row: f"{model_label(row['model'])}\n{method_label(row['method'])}", axis=1)
    heat = temporal.set_index("label")[
        ["recent_0_15_min_share", "mid_15_30_min_share", "older_30_60_min_share"]
    ]
    heat.columns = ["0-15 min", "15-30 min", "30-60 min"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.imshow(heat.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels(heat.columns)
    ax.set_xlabel("Traffic evidence time window before the alarm")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#08306B")
    fig.colorbar(image, ax=ax, label="Mean attribution share")
    ax.set_title("Temporal attribution evidence across all alarm episodes")
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "temporal_attribution_heatmap_all_episodes.png",
        "xai_alarm_evidence_metrics.csv",
        "XAI evidence figures",
        "XAI temporal evidence",
        "Mean attribution share by temporal recency for each model-method combination.",
        "Recent evidence indicates that alarms are mainly explained by traffic context near the alarm timestamp.",
    )
    return fig


def feature_family_share(ctx: ReportContext) -> plt.Figure:
    family = ctx.xai_evidence.groupby(["model", "method"])[
        ["speed_attribution_share", "occupancy_attribution_share"]
    ].mean().reset_index()
    family["sort_model"] = family["model"].map({model: i for i, model in enumerate(MODEL_ORDER)})
    family = family.sort_values(["sort_model", "method"])
    labels = family.apply(lambda row: f"{model_label(row['model'])}\n{method_label(row['method'])}", axis=1)
    x_pos = np.arange(len(family))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x_pos, family["speed_attribution_share"], label="Speed contribution", color="#2C7FB8")
    ax.bar(x_pos, family["occupancy_attribution_share"], bottom=family["speed_attribution_share"], label="Occupancy contribution", color="#D7301F")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean attribution share")
    ax.set_title("Speed and occupancy contribution by model-method")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "feature_family_share_by_model_method.png",
        "xai_alarm_evidence_metrics.csv",
        "XAI evidence figures",
        "XAI feature evidence",
        "Mean speed and occupancy attribution share by model-method across all alarm episode representatives.",
        "Because the model input contains only speed and occupancy, temporal structure is often more informative than the feature family alone.",
    )
    return fig


def explanation_concentration(ctx: ReportContext) -> plt.Figure:
    groups = []
    labels = []
    for (model, method), group in ctx.xai_evidence.groupby(["model", "method"]):
        groups.append(group["top3_attribution_concentration"].dropna())
        labels.append(f"{model_label(model)}\n{method_label(method)}")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Explanation concentration")
    ax.set_title("Top-3 attribution concentration across alarm episodes")
    ax.tick_params(axis="x", rotation=28, labelsize=8.5)
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "explanation_concentration_by_model_method.png",
        "xai_alarm_evidence_metrics.csv",
        "XAI evidence figures",
        "XAI evidence quality",
        "Distribution of top-3 attribution concentration by model-method.",
        "Higher concentration means the explanation can be summarized using fewer dominant pieces of traffic evidence.",
    )
    return fig


def score_margin_vs_concentration(ctx: ReportContext) -> plt.Figure:
    joined = ctx.xai_evidence.merge(
        ctx.xai_targets[["model", "fold", "representative_row_id", "score_margin", "model_agreement_count"]],
        left_on=["model", "fold", "row_id"],
        right_on=["model", "fold", "representative_row_id"],
        how="left",
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODEL_ORDER:
        group = joined[joined["model"] == model]
        ax.scatter(
            group["score_margin"].clip(-0.02, 0.2),
            group["top3_attribution_concentration"],
            s=16,
            alpha=0.35,
            color=MODEL_COLORS[model],
            label=model_label(model),
        )
    ax.set_xlabel("Score margin above threshold")
    ax.set_ylabel("Explanation concentration")
    ax.set_title("Alarm strength versus explanation concentration")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "score_margin_vs_explanation_concentration.png",
        "xai_alarm_evidence_metrics.csv; xai_alarm_episode_targets.csv",
        "XAI evidence figures",
        "XAI evidence quality",
        "Scatter plot linking score margin above threshold with explanation concentration.",
        "Strong alarms and concentrated explanations are different evidence-quality dimensions; both matter for triage.",
    )
    return fig


def shap_lime_agreement(ctx: ReportContext) -> plt.Figure:
    models = ["random_forest", "xgboost", "mlp"]
    data = [
        ctx.xai_method_agreement[ctx.xai_method_agreement["model"] == model]["top_feature_overlap_score"]
        for model in models
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot(data, labels=[model_label(model) for model in models], showfliers=False)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Agreement score")
    ax.set_title("SHAP/LIME-style top-feature agreement by model")
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "shap_lime_agreement_by_model.png",
        "xai_method_agreement_all_alarm_episodes.csv",
        "XAI evidence figures",
        "XAI method agreement",
        "Distribution of SHAP/LIME-style top-feature overlap for models with two explanation methods.",
        "Higher agreement means the explanation is more stable across explanation methods for the same alarm case.",
    )
    return fig


def cross_model_agreement_figures(ctx: ReportContext) -> list[plt.Figure]:
    pairs = ctx.xai_cross_model.copy()
    score_matrix = pd.DataFrame(index=[model_label(m) for m in MODEL_ORDER], columns=[model_label(m) for m in MODEL_ORDER], dtype=float)
    grouped = pairs.groupby(["left_model", "right_model"])["top_feature_overlap_score"].mean().reset_index()
    for _, row in grouped.iterrows():
        left = model_label(row["left_model"])
        right = model_label(row["right_model"])
        score_matrix.loc[left, right] = row["top_feature_overlap_score"]
        score_matrix.loc[right, left] = row["top_feature_overlap_score"]
    matrix_values = score_matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(matrix_values, 1.0)
    score_matrix = pd.DataFrame(
        matrix_values,
        index=score_matrix.index,
        columns=score_matrix.columns,
    )
    fig1, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(score_matrix.values.astype(float), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(score_matrix.columns)))
    ax.set_xticklabels(score_matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(score_matrix.index)))
    ax.set_yticklabels(score_matrix.index)
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            ax.text(j, i, f"{score_matrix.values[i, j]:.2f}", ha="center", va="center", color="#08306B")
    fig1.colorbar(image, ax=ax, label="Agreement score")
    ax.set_title("Cross-model explanation agreement")
    fig1.tight_layout()
    save_figure(
        ctx,
        fig1,
        "cross_model_agreement_heatmap.png",
        "xai_cross_model_agreement_all_alarm_episodes.csv",
        "XAI evidence figures",
        "Cross-model XAI agreement",
        "Average top-feature agreement between model-family explanations on shared alarm cases.",
        "Cross-model agreement shows whether different model families point to similar traffic evidence.",
    )

    pairs["pair"] = pairs.apply(lambda row: f"{model_label(row['left_model'])} vs {model_label(row['right_model'])}", axis=1)
    levels = ["consistent", "partially_consistent", "conflicting"]
    counts = pairs.groupby(["pair", "explanation_agreement_level"]).size().unstack(fill_value=0).reindex(columns=levels, fill_value=0)
    fig2, ax = plt.subplots(figsize=(11.5, 6))
    bottom = np.zeros(len(counts))
    colors = {"consistent": "#41AB5D", "partially_consistent": "#FEC44F", "conflicting": "#D7301F"}
    totals = counts.sum(axis=1).replace(0, np.nan)
    for level in levels:
        values = counts[level].values
        bars = ax.bar(counts.index, values, bottom=bottom, label=clean_level(level), color=colors[level])
        for index, (bar, value, base) in enumerate(zip(bars, values, bottom)):
            if value <= 0:
                continue
            share = value / totals.iloc[index]
            if share >= 0.08:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{share:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#1f1f1f",
                )
        bottom += values
    ax.set_title("Cross-model explanation agreement levels", pad=26)
    ax.text(
        0.5,
        1.02,
        "Agreement combines top-feature overlap and contribution-direction agreement",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#525252",
    )
    ax.set_ylabel("Shared alarm representative rows")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(
        title="Explanation level",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.text(
        0.0,
        -0.28,
        "Consistent: similar important lagged features and direction. Partially consistent: some shared evidence. Conflicting: little overlap or different direction.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#525252",
        wrap=True,
    )
    fig2.subplots_adjust(left=0.09, right=0.78, top=0.82, bottom=0.30)
    save_figure(
        ctx,
        fig2,
        "cross_model_agreement_level_summary.png",
        "xai_cross_model_agreement_all_alarm_episodes.csv",
        "XAI evidence figures",
        "Cross-model XAI agreement",
        "Counts of consistent, partially consistent, and conflicting cross-model explanation pairs.",
        "Agreement-level summaries identify where model families provide similar or conflicting support for alarms.",
    )
    return [fig1, fig2]


def strong_vs_borderline_profiles(ctx: ReportContext) -> plt.Figure:
    joined = ctx.xai_evidence.merge(
        ctx.xai_targets[
            [
                "model",
                "fold",
                "stream_id",
                "representative_row_id",
                "alarm_start",
                "alarm_end",
                "score",
                "threshold",
                "score_margin",
                "model_agreement_count",
            ]
        ],
        left_on=["model", "fold", "row_id"],
        right_on=["model", "fold", "representative_row_id"],
        how="left",
    )
    cases = joined.groupby(["model", "fold", "row_id"]).agg(
        stream_id=("stream_id", "first"),
        alarm_start=("alarm_start", "first"),
        alarm_end=("alarm_end", "first"),
        score=("score", "first"),
        threshold=("threshold", "first"),
        score_margin=("score_margin", "first"),
        model_agreement_count=("model_agreement_count", "first"),
        recent_share=("recent_0_15_min_share", "mean"),
        concentration=("top3_attribution_concentration", "mean"),
        top_features=("top_10_features", "first"),
    ).reset_index().dropna(subset=["score_margin"])
    cases["strength"] = (
        cases["score_margin"].rank(pct=True)
        + cases["model_agreement_count"].rank(pct=True)
        + cases["recent_share"].rank(pct=True)
        + cases["concentration"].rank(pct=True)
    )
    strong = cases.sort_values("strength", ascending=False).iloc[0]
    borderline = cases[(cases["score_margin"] > 0) & (cases["model_agreement_count"] <= 2)].sort_values(
        ["score_margin", "concentration"]
    ).iloc[0]

    def top_features(value: str) -> str:
        return "\n".join(f"- {feature_label(item.strip())}" for item in str(value).split(",")[:3])

    def case_timestamp(row: pd.Series) -> pd.Timestamp:
        match = ctx.training.loc[ctx.training["row_id"] == int(row["row_id"]), "timestamp"]
        if match.empty:
            return pd.NaT
        return pd.Timestamp(match.iloc[0])

    def normalized(series: pd.Series) -> pd.Series:
        spread = series.max() - series.min()
        if spread <= 0:
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / spread

    def plot_alarm_context(ax: plt.Axes, row: pd.Series) -> None:
        stream_rows = ctx.training[ctx.training["stream_id"] == int(row["stream_id"])].copy()
        stream_rows = stream_rows.sort_values("timestamp").reset_index(drop=True)
        position = stream_rows.index[stream_rows["row_id"] == int(row["row_id"])]
        if len(position) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "Traffic context unavailable", ha="center", va="center")
            return
        center = int(position[0])
        start = max(0, center - 12)
        end = min(len(stream_rows), center + 13)
        window = stream_rows.iloc[start:end].copy()
        window["speed_norm"] = normalized(window["speed_smoothed"])
        window["occ_norm"] = normalized(window["occ_smoothed"])
        rep_time = pd.Timestamp(window.loc[window["row_id"] == int(row["row_id"]), "timestamp"].iloc[0])
        alarm_start = pd.Timestamp(row["alarm_start"])
        alarm_end = pd.Timestamp(row["alarm_end"])
        ax.plot(window["timestamp"], window["speed_norm"], color="#2C7FB8", lw=2.0, label="Speed")
        ax.plot(window["timestamp"], window["occ_norm"], color="#D7301F", lw=2.0, label="Occupancy")
        ax.axvspan(alarm_start, alarm_end, color="#FEC44F", alpha=0.25, label="Alarm episode")
        ax.axvline(rep_time, color="#252525", lw=1.4, linestyle="--", label="Representative row")
        ax.set_title("Traffic context around alarm", fontsize=10, pad=8)
        ax.set_ylabel("Normalized value")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
        ax.text(
            0.99,
            -0.28,
            "Values normalized inside this local window.",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
            color="#525252",
        )

    fig = plt.figure(figsize=(14, 9.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.95, 1.45],
        height_ratios=[1, 1],
        hspace=0.50,
        wspace=0.22,
    )
    profiles = [
        (
            "Strong review-worthy alarm",
            strong,
            "#E5F5E0",
            "#238B45",
            "Evidence is strong, recent, and concentrated.\nReview camera, map, or sensor context.",
        ),
        (
            "Borderline lower-priority alarm",
            borderline,
            "#FEF0D9",
            "#D95F0E",
            "Evidence is weaker and closer to threshold.\nKeep visible, but prioritize lower unless context worsens.",
        ),
    ]
    for index, (title, row, color, accent, note) in enumerate(profiles):
        ax_card = fig.add_subplot(grid[index, 0])
        ax_plot = fig.add_subplot(grid[index, 1])
        ax_card.axis("off")
        card = patches.FancyBboxPatch(
            (0.02, 0.03),
            0.96,
            0.92,
            boxstyle="round,pad=0.025,rounding_size=0.025",
            facecolor=color,
            edgecolor=accent,
            linewidth=1.4,
        )
        ax_card.add_patch(card)
        ax_card.add_patch(
            patches.Rectangle((0.02, 0.86), 0.96, 0.09, facecolor=accent, alpha=0.95)
        )
        rep_time = case_timestamp(row)
        ax_card.text(
            0.06,
            0.905,
            title,
            ha="left",
            va="center",
            fontsize=10.5,
            color="white",
            fontweight="bold",
        )
        metrics = (
            f"Model: {model_label(row['model'])}   Fold: {int(row['fold'])}\n"
            f"Stream: {int(row['stream_id'])}   Alarm time: {rep_time:%m-%d %H:%M}\n"
            f"Score: {row['score']:.3f}   Threshold: {row['threshold']:.3f}\n"
            f"Margin: {row['score_margin']:.3f}   Models alarming: {int(row['model_agreement_count'])}/4"
        )
        ax_card.text(0.07, 0.80, metrics, ha="left", va="top", fontsize=8.0, linespacing=1.28)
        ax_card.text(
            0.07,
            0.55,
            f"Evidence quality: recent {row['recent_share']:.2f} | concentration {row['concentration']:.2f}",
            ha="left",
            va="top",
            fontsize=8.4,
            color="#252525",
            fontweight="bold",
        )
        ax_card.text(
            0.07,
            0.44,
            f"Top evidence\n{top_features(row['top_features'])}",
            ha="left",
            va="top",
            fontsize=8.1,
            linespacing=1.18,
        )
        ax_card.text(
            0.07,
            0.19,
            f"Operator note\n{note}",
            ha="left",
            va="top",
            fontsize=8.1,
            linespacing=1.12,
            color="#252525",
        )
        plot_alarm_context(ax_plot, row)
    fig.suptitle("Operator-facing alarm evidence cards", y=0.98, fontsize=14)
    save_figure(
        ctx,
        fig,
        "strong_vs_borderline_alarm_profiles.png",
        "xai_alarm_evidence_metrics.csv; xai_alarm_episode_targets.csv",
        "Operator-facing triage figures",
        "Operator triage examples",
        "Side-by-side operator-facing profiles for a strong and borderline alarm.",
        "The card wording avoids ground-truth label claims and focuses on alarm evidence quality.",
    )
    return fig


def alarm_evidence_profile_summary(ctx: ReportContext) -> plt.Figure:
    nodes = [
        ("Alarm score", "How abnormal the model output is"),
        ("Threshold margin", "How far above the decision threshold"),
        ("Model agreement", "How many model families also alarm"),
        ("XAI recentness", "Whether evidence is from recent traffic context"),
        ("XAI concentration", "Whether evidence is focused or diffuse"),
        ("Method agreement", "Whether explanation methods agree"),
        ("Operator priority", "Review priority and supporting note"),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(nodes))
    for index, (x_pos, (title, subtitle)) in enumerate(zip(xs, nodes)):
        face = "#E5F5E0" if index == len(nodes) - 1 else "#F7FBFF"
        box = patches.FancyBboxPatch((x_pos - 0.055, 0.38), 0.11, 0.28, boxstyle="round,pad=0.02,rounding_size=0.015", facecolor=face, edgecolor="#34495E")
        ax.add_patch(box)
        ax.text(x_pos, 0.55, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x_pos, 0.43, wrap_text(subtitle, 17), ha="center", va="center", fontsize=7.5)
        if index < len(nodes) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.065, 0.52), xytext=(x_pos + 0.065, 0.52), arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#34495E"})
    ax.text(0.5, 0.18, "XAI supports operator judgment under uncertainty; it explains model behavior, not physical causality.", ha="center", fontsize=10, color="#525252")
    ax.set_title("Alarm evidence profile used for operator triage")
    save_figure(
        ctx,
        fig,
        "alarm_evidence_profile_summary.png",
        "alarm_episode_triage.csv; xai_alarm_evidence_metrics.csv; xai_method_agreement_all_alarm_episodes.csv",
        "Operator-facing triage figures",
        "Operator triage workflow",
        "Compact diagram of how alarm strength and XAI evidence quality feed into operator priority.",
        "The final layer combines score strength, model consensus, temporal evidence, explanation concentration, and method agreement.",
    )
    return fig


def shapley_value_context(ctx: ReportContext) -> plt.Figure:
    """Explain where local SHAP/LIME-style attribution values come from."""

    boxes = [
        (
            "Alarm representative row",
            "One timestamp selected from an alarm episode, usually the peak-score row.",
        ),
        (
            "Lookback input window",
            "Previous 60 minutes of smoothed Speed and Occupancy values.",
        ),
        (
            "Saved trained model",
            "Random Forest, XGBoost, MLP, or LSTM produces an anomaly score.",
        ),
        (
            "Reference behaviour",
            "Baseline/background samples describe normal model input context.",
        ),
        (
            "Attribution values",
            "SHAP/LIME-style values assign contribution to each lagged input.",
        ),
        (
            "Evidence summary",
            "Recentness, concentration, agreement, and operator priority.",
        ),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.axis("off")
    xs = [0.10, 0.30, 0.50, 0.50, 0.70, 0.90]
    ys = [0.68, 0.68, 0.68, 0.30, 0.68, 0.68]
    for index, ((title, subtitle), x_pos, y_pos) in enumerate(zip(boxes, xs, ys)):
        face = "#F7FBFF" if index != 4 else "#FFF7BC"
        box = patches.FancyBboxPatch(
            (x_pos - 0.075, y_pos - 0.12),
            0.15,
            0.24,
            boxstyle="round,pad=0.02,rounding_size=0.014",
            facecolor=face,
            edgecolor="#34495E",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x_pos, y_pos + 0.045, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x_pos, y_pos - 0.045, wrap_text(subtitle, 22), ha="center", va="center", fontsize=7.8)
    arrows = [
        ((0.175, 0.68), (0.225, 0.68)),
        ((0.375, 0.68), (0.425, 0.68)),
        ((0.575, 0.68), (0.625, 0.68)),
        ((0.50, 0.42), (0.50, 0.56)),
        ((0.775, 0.68), (0.825, 0.68)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#34495E"})
    ax.text(
        0.50,
        0.12,
        "An attribution value is not a traffic measurement. It is a model-explanation value: how much a lagged input helped move the model output away from its reference behaviour for this alarm row.",
        ha="center",
        va="center",
        fontsize=10,
        color="#525252",
        wrap=True,
    )
    ax.set_title("How local XAI attribution values are produced and interpreted")
    save_figure(
        ctx,
        fig,
        "shapley_value_context.png",
        "xai_local_explanations_all_alarm_episodes.csv; xai_alarm_evidence_metrics.csv",
        "XAI evidence figures",
        "XAI interpretation",
        "Conceptual diagram showing how an alarm representative row becomes local attribution values and evidence-quality metrics.",
        "This figure gives report context for SHAP/LIME-style values before showing attribution bars.",
    )
    return fig


def _lagged_feature_values(
    training: pd.DataFrame,
    local: pd.DataFrame,
) -> pd.Series:
    """Return the traffic value behind each lagged explanation row."""

    needed = local[["row_id", "feature"]].copy()
    needed["traffic_signal"] = np.where(
        needed["feature"].str.startswith("speed"),
        "speed_smoothed",
        "occ_smoothed",
    )
    needed["lag_steps"] = needed["feature"].str.extract(r"t_minus_(\d+)").astype(int)
    lookup = training[["stream_id", "row_id", "speed_smoothed", "occ_smoothed"]].copy()
    rows = training[["stream_id", "row_id"]].copy()
    rows["position"] = rows.groupby("stream_id").cumcount()
    lookup = lookup.merge(rows, on=["stream_id", "row_id"], how="left")
    local_rows = local[["stream_id", "row_id"]].drop_duplicates()
    local_rows = local_rows.merge(rows, on=["stream_id", "row_id"], how="left")
    needed = needed.merge(local_rows, on="row_id", how="left")
    needed["lag_position"] = needed["position"] - needed["lag_steps"]
    values = needed.merge(
        lookup.rename(columns={"position": "lag_position"}),
        on=["stream_id", "lag_position"],
        how="left",
        suffixes=("", "_lag"),
    )
    return np.where(
        values["traffic_signal"] == "speed_smoothed",
        values["speed_smoothed"],
        values["occ_smoothed"],
    )


def shap_style_summary_beeswarm(ctx: ReportContext) -> plt.Figure:
    """Create a SHAP-style summary plot across all alarm episode explanations."""

    local_path = config.TABLES_DIR / "xai_local_explanations_all_alarm_episodes.csv"
    local = pd.read_csv(local_path)
    shap_like = local[
        local["method"].isin(["treeshap", "deepshap", "sequence_deepshap"])
    ].copy()
    if shap_like.empty:
        raise FileNotFoundError("No SHAP-style local explanation rows available.")

    totals = shap_like.groupby("case_id")["abs_attribution"].transform("sum")
    shap_like = shap_like[totals > 0].copy()
    shap_like["signed_explanation_share"] = shap_like["attribution"] / totals
    shap_like["abs_explanation_share"] = shap_like["abs_attribution"] / totals
    top_features = (
        shap_like.groupby("feature")["signed_explanation_share"]
        .apply(lambda values: values.abs().mean())
        .sort_values(ascending=False)
        .head(14)
        .index.tolist()
    )
    plot_data = shap_like[shap_like["feature"].isin(top_features)].copy()
    plot_data["feature_value"] = _lagged_feature_values(ctx.training, plot_data)
    plot_data["feature_value_scaled"] = plot_data.groupby("feature")[
        "feature_value"
    ].transform(
        lambda values: (
            (values - values.min()) / (values.max() - values.min())
            if values.max() > values.min()
            else 0.5
        )
    )
    is_speed = plot_data["feature"].str.startswith("speed")
    domain_direction = np.where(
        is_speed,
        1 - (2 * plot_data["feature_value_scaled"]),
        (2 * plot_data["feature_value_scaled"]) - 1,
    )
    plot_data["domain_aligned_alarm_evidence"] = (
        plot_data["abs_explanation_share"] * domain_direction
    )
    plot_data["mean_abs_share"] = plot_data.groupby("feature")[
        "signed_explanation_share"
    ].transform(lambda values: values.abs().mean())
    plot_data = plot_data.sort_values(
        ["mean_abs_share", "feature"],
        ascending=[False, True],
    )
    feature_order = plot_data["feature"].drop_duplicates().tolist()
    feature_order.reverse()

    rng = np.random.default_rng(42)
    max_points_per_feature = 900
    sampled_groups = []
    for feature, group in plot_data.groupby("feature", sort=False):
        if len(group) > max_points_per_feature:
            sampled_groups.append(
                group.sample(max_points_per_feature, random_state=42)
            )
        else:
            sampled_groups.append(group)
    sampled = pd.concat(sampled_groups, ignore_index=True)
    y_positions = {feature: index for index, feature in enumerate(feature_order)}
    sampled["y_pos"] = sampled["feature"].map(y_positions)
    sampled["jitter"] = rng.normal(0, 0.075, len(sampled))

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    scatter = ax.scatter(
        sampled["domain_aligned_alarm_evidence"],
        sampled["y_pos"] + sampled["jitter"],
        c=sampled["feature_value_scaled"],
        cmap="plasma",
        s=11,
        alpha=0.55,
        linewidths=0,
    )
    ax.axvline(0, color="#252525", linewidth=1)
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels([feature_label(feature) for feature in feature_order])
    ax.set_xlabel("Domain-aligned alarm evidence share")
    ax.set_ylabel("Lagged input feature")
    ax.set_title("Domain-aligned SHAP-style alarm evidence across alarm episodes")
    ax.text(
        0.02,
        1.02,
        "less alarm-supporting",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#525252",
    )
    ax.text(
        0.98,
        1.02,
        "more alarm-supporting",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#525252",
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Feature value within feature range")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    ax.text(
        0.01,
        -0.12,
        "Dots are alarm episode representative rows. Right of zero means congestion-like alarm evidence: lower Speed or higher Occupancy. Left of zero means the feature value is less consistent with alarm evidence.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#525252",
        wrap=True,
    )
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "shap_style_summary_beeswarm_all_episodes.png",
        "xai_local_explanations_all_alarm_episodes.csv; training_dataset.csv",
        "XAI evidence figures",
        "XAI population-level summary",
        "Domain-aligned SHAP-style beeswarm summary showing alarm evidence across all alarm episode representative rows.",
        "This view converts attribution magnitude into a traffic-domain direction, so low Speed and high Occupancy are shown as alarm-supporting evidence.",
    )
    return fig


def local_treeshap_value_example(ctx: ReportContext) -> plt.Figure:
    """Show a real local TreeSHAP attribution example with readable labels."""

    local_path = config.TABLES_DIR / "xai_local_explanations_all_alarm_episodes.csv"
    local = pd.read_csv(local_path)
    targets = ctx.xai_targets[
        [
            "representative_row_id",
            "model",
            "fold",
            "score_margin",
            "model_agreement_count",
            "true_label",
            "case_type",
        ]
    ]
    candidates = local[
        (local["method"] == "treeshap")
        & (local["model"].isin(["random_forest", "xgboost"]))
        & (local["feature"].str.startswith("speed_smoothed"))
        & (local["attribution"] > 0)
        & (local["rank"] <= 3)
    ].merge(
        targets,
        left_on=["row_id", "model", "fold"],
        right_on=["representative_row_id", "model", "fold"],
        how="left",
    )
    if candidates.empty:
        raise FileNotFoundError("No positive speed TreeSHAP example rows available.")
    candidates = candidates.sort_values(
        ["model_agreement_count", "score_margin", "abs_attribution"],
        ascending=[False, False, False],
    )
    case_id = str(candidates["case_id"].iloc[0])
    case = local[
        (local["case_id"] == case_id)
        & (local["method"] == "treeshap")
    ].nsmallest(10, "rank").copy()
    row_id = int(case["row_id"].iloc[0])
    model_name = str(case["model"].iloc[0])
    target_match = ctx.xai_targets[
        (ctx.xai_targets["representative_row_id"].astype(int) == row_id)
        & (ctx.xai_targets["model"] == model_name)
    ]
    if not target_match.empty:
        target = target_match.iloc[0]
        offline_label = (
            "Offline evaluation: overlapped labeled incident"
            if int(target["true_label"]) == 1
            else "Offline evaluation: no labeled-incident overlap"
        )
        score_text = (
            f"Score margin: {float(target['score_margin']):.3f}; "
            f"models alarming: {int(target['model_agreement_count'])}/4"
        )
    else:
        offline_label = "Offline evaluation: not available"
        score_text = "Score margin/model agreement unavailable"
    case["readable_feature"] = case["feature"].map(feature_label)
    case = case.sort_values("attribution")
    colors = np.where(case["attribution"] >= 0, "#D7301F", "#2C7FB8")
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh(case["readable_feature"], case["attribution"], color=colors)
    ax.axvline(0, color="#525252", linewidth=1)
    model = model_label(str(case["model"].iloc[0]))
    ax.set_title(f"Local TreeSHAP example: recent Speed supports an alarm ({model})")
    ax.text(
        0.0,
        1.04,
        f"{offline_label}. {score_text}.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#525252",
    )
    ax.set_xlabel("Attribution value")
    ax.set_ylabel("Lagged input feature")
    legend_handles = [
        patches.Patch(color="#D7301F", label="Supports alarm direction"),
        patches.Patch(color="#2C7FB8", label="Supports normal direction"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(
        ctx,
        fig,
        "local_treeshap_value_example.png",
        "xai_local_explanations_all_alarm_episodes.csv",
        "XAI evidence figures",
        "XAI interpretation",
        "A real local TreeSHAP example where recent Speed has a positive contribution to a tree-model alarm, with offline evaluation context.",
        "This example is easier to present than the beeswarm because it shows one concrete model decision and how a recent speed feature supports the alarm.",
    )
    save_figure(
        ctx,
        fig,
        "speed_positive_shap_example.png",
        "xai_local_explanations_all_alarm_episodes.csv",
        "XAI evidence figures",
        "XAI interpretation",
        "A real local TreeSHAP example where recent Speed has a positive contribution to a tree-model alarm.",
        "Use this presentation-friendly figure to explain how a lagged speed value can push one alarm decision upward.",
    )
    return fig


FIGURE_BUILDERS: list[Callable[[ReportContext], plt.Figure | list[plt.Figure]]] = [
    project_pipeline_overview,
    incident_window_protocol,
    stream_level_cv_diagram,
    dataset_label_window_summary,
    model_performance_figures,
    fold_performance_stability,
    threshold_score_overview,
    example_alarm_timeline,
    timestamp_to_alarm_episode_grouping,
    alarm_episode_summary,
    xai_coverage_overview,
    shapley_value_context,
    shap_style_summary_beeswarm,
    local_treeshap_value_example,
    temporal_attribution_heatmap,
    feature_family_share,
    explanation_concentration,
    score_margin_vs_concentration,
    shap_lime_agreement,
    cross_model_agreement_figures,
    strong_vs_borderline_profiles,
    alarm_evidence_profile_summary,
]


def build_all_report_figures(ctx: ReportContext | None = None) -> ReportContext:
    """Generate every report figure and export the manifest."""

    context = ctx or load_context()
    context.manifest_rows.clear()
    for builder in FIGURE_BUILDERS:
        result = builder(context)
        if isinstance(result, list):
            for fig in result:
                plt.close(fig)
        else:
            plt.close(result)
    manifest = pd.DataFrame([row.__dict__ for row in context.manifest_rows])
    manifest.to_csv(MANIFEST_PATH, index=False)
    return context
