"""Train RF, XGBoost, MLP, and LSTM baseline-prediction detectors."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from traffic_incident_xai import config
from traffic_incident_xai.data_pipeline import consolidate_dataset

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - surfaced by CLI validation.
    XGBRegressor = None  # type: ignore[assignment]

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - surfaced by CLI validation.
    torch = None  # type: ignore[assignment]
    class _MissingTorchModule:
        """Minimal placeholder so this module remains importable."""

        Module = object

    nn = _MissingTorchModule()  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TrainingResult:
    """Serializable training output summary."""

    model_name: str
    fold: int
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    tp: int
    fp: int
    fn: int
    tn: int
    train_windows: int
    calibration_windows: int
    test_windows: int
    model_path: str
    score_path: str
    run_mode: str
    threshold_strategy: str
    epochs_trained: int | None = None


class TabularMlpRegressor(nn.Module):  # type: ignore[misc]
    """Tabular MLP next-step regressor for flattened lookback windows."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict the next speed/occupancy vector."""

        return self.network(features)


class StationLstmRegressor(nn.Module):  # type: ignore[misc]
    """LSTM next-step regressor with learned station-id embeddings."""

    def __init__(
        self,
        n_stations: int,
        n_features: int,
        embedding_dim: int = 8,
        hidden_dim: int = 48,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_stations, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=n_features + embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, sequence: torch.Tensor, station: torch.Tensor) -> torch.Tensor:
        """Predict the next speed/occupancy vector."""

        station_embedding = self.embedding(station)
        repeated = station_embedding.unsqueeze(1).repeat(1, sequence.size(1), 1)
        joined = torch.cat([sequence, repeated], dim=2)
        hidden, _ = self.lstm(joined)
        return self.output(hidden[:, -1, :])


def torch_device() -> "torch.device":
    """Return CUDA when available, otherwise CPU."""

    if torch is None:
        raise ImportError("torch is required for neural model training.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cpu_state_dict(model: nn.Module) -> dict[str, "torch.Tensor"]:
    """Return a CPU copy of a model state dict for portable checkpoints."""

    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def load_dataset(max_streams: int | None = None) -> pd.DataFrame:
    """Load existing consolidated data or rebuild it from source CSV files."""

    dataset_path = config.PROCESSED_DIR / "training_dataset.csv"
    if not dataset_path.exists() or max_streams is not None:
        stream_ids = list(config.STREAM_IDS)
        if max_streams is not None:
            stream_ids = stream_ids[:max_streams]
        return consolidate_dataset(stream_ids=stream_ids, output_path=dataset_path)
    frame = pd.read_csv(dataset_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def make_windows(
    frame: pd.DataFrame,
    lookback: int = config.LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Create sequential windows and next-step targets within each stream/split."""

    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    stations: list[int] = []
    metadata: list[dict[str, object]] = []
    group_columns = ["stream_id", "split"]
    for (_, split), group in frame.groupby(group_columns, sort=False):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        values = ordered.loc[:, config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        station_id = int(ordered["station_id"].iloc[0])
        for index in range(lookback, len(ordered)):
            sequences.append(values[index - lookback : index])
            targets.append(values[index])
            stations.append(station_id)
            metadata.append(
                {
                    "row_id": int(ordered["row_id"].iloc[index]),
                    "stream_id": int(ordered["stream_id"].iloc[index]),
                    "source_file": str(ordered["source_file"].iloc[index]),
                    "fold": int(ordered["fold"].iloc[index]),
                    "split": split,
                    "row_in_split": int(ordered["row_in_split"].iloc[index]),
                    "station_id": station_id,
                    "timestamp": ordered["timestamp"].iloc[index],
                    "is_incident": int(ordered["is_incident"].iloc[index]),
                }
            )
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(stations, dtype=np.int64),
        pd.DataFrame(metadata),
    )


def flatten_windows(sequences: np.ndarray) -> np.ndarray:
    """Flatten lookback windows for tabular models."""

    return sequences.reshape(sequences.shape[0], -1)


def anomaly_scores(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Compute bivariate residual anomaly scores."""

    return np.sqrt(np.mean(np.square(predicted - actual), axis=1))


def calibrate_threshold(scores: np.ndarray) -> float:
    """Select tau-star from baseline-only training residuals."""

    return float(np.quantile(scores, config.THRESHOLD_QUANTILE))


def split_fit_calibration_masks(
    metadata: pd.DataFrame,
    fold: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split non-held-out baseline windows into fit and calibration windows."""

    baseline_mask = (metadata["split"] == "train") & (metadata["fold"] != fold)
    heldout_mask = (metadata["split"] == "test") & (metadata["fold"] == fold)
    cutoff = int(config.TRAIN_ROWS * (1.0 - config.CALIBRATION_FRACTION))
    calibration_mask = baseline_mask & (metadata["row_in_split"] >= cutoff)
    fit_mask = baseline_mask & (metadata["row_in_split"] < cutoff)
    return fit_mask, calibration_mask, heldout_mask


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    """Evaluate binary alarms at the calibrated threshold."""

    predictions = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    roc_auc: float | None
    pr_auc: float | None
    if len(np.unique(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
    else:
        roc_auc = None
        pr_auc = None
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def save_predictions(
    model_dir: Path,
    metadata: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    model_name: str,
    fold: int,
) -> Path:
    """Persist anomaly scores and thresholded alarms."""

    output = metadata.copy()
    output["model"] = model_name
    output["score"] = scores
    output["normalized_score"] = scores / (scores + threshold + 1e-12)
    output["threshold"] = threshold
    output["threshold_margin"] = scores - threshold
    output["prediction"] = (scores > threshold).astype(int)
    output["true_label"] = output["is_incident"].astype(int)
    output["case_type"] = np.select(
        [
            (output["prediction"] == 1) & (output["true_label"] == 1),
            (output["prediction"] == 1) & (output["true_label"] == 0),
            (output["prediction"] == 0) & (output["true_label"] == 1),
        ],
        ["true_positive", "false_positive", "missed_incident"],
        default="true_negative",
    )
    score_path = model_dir / f"{model_name}_fold_{fold}_scores.csv"
    output.to_csv(score_path, index=False)
    return score_path


def train_tabular_fold(
    model_name: str,
    factory: Callable[[], object],
    sequences: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
    fold: int,
    run_mode: str,
    output_root: Path,
) -> TrainingResult:
    """Train one tabular model for a held-out fold."""

    model_dir = output_root / model_name / f"fold_{fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_mask, calibration_mask, test_mask = split_fit_calibration_masks(
        metadata,
        fold,
    )
    x_train = flatten_windows(sequences[fit_mask.to_numpy()])
    y_train = targets[fit_mask.to_numpy()]
    x_calibration = flatten_windows(sequences[calibration_mask.to_numpy()])
    y_calibration = targets[calibration_mask.to_numpy()]
    x_test = flatten_windows(sequences[test_mask.to_numpy()])
    y_test = targets[test_mask.to_numpy()]
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    x_calibration_scaled = x_scaler.transform(x_calibration)
    y_calibration_scaled = y_scaler.transform(y_calibration)
    x_test_scaled = x_scaler.transform(x_test)
    y_test_scaled = y_scaler.transform(y_test)

    model = factory()
    model.fit(x_train_scaled, y_train_scaled)
    calibration_scores = anomaly_scores(
        model.predict(x_calibration_scaled),
        y_calibration_scaled,
    )
    threshold = calibrate_threshold(calibration_scores)
    test_scores = anomaly_scores(model.predict(x_test_scaled), y_test_scaled)
    metrics = evaluate_scores(
        metadata.loc[test_mask, "is_incident"].to_numpy(dtype=int),
        test_scores,
        threshold,
    )

    model_path = model_dir / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "lookback": config.LOOKBACK,
            "feature_columns": config.FEATURE_COLUMNS,
            "threshold": threshold,
            "run_mode": run_mode,
            "threshold_strategy": "baseline_calibration_quantile",
            "threshold_quantile": config.THRESHOLD_QUANTILE,
        },
        model_path,
    )
    score_path = save_predictions(
        model_dir,
        metadata.loc[test_mask].reset_index(drop=True),
        test_scores,
        threshold,
        model_name,
        fold,
    )
    return TrainingResult(
        model_name=model_name,
        fold=fold,
        threshold=threshold,
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        tp=int(metrics["tp"]),
        fp=int(metrics["fp"]),
        fn=int(metrics["fn"]),
        tn=int(metrics["tn"]),
        train_windows=int(fit_mask.sum()),
        calibration_windows=int(calibration_mask.sum()),
        test_windows=int(test_mask.sum()),
        model_path=str(model_path),
        score_path=str(score_path),
        run_mode=run_mode,
        threshold_strategy="baseline_calibration_quantile",
    )


def train_mlp_fold(
    sequences: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
    fold: int,
    epochs: int,
    batch_size: int,
    run_mode: str,
    output_root: Path,
) -> TrainingResult:
    """Train the PyTorch MLP for a held-out fold."""

    if torch is None:
        raise ImportError("torch is required for MLP training.")
    model_name = "mlp"
    model_dir = output_root / model_name / f"fold_{fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_mask, calibration_mask, test_mask = split_fit_calibration_masks(
        metadata,
        fold,
    )
    x_train = flatten_windows(sequences[fit_mask.to_numpy()])
    y_train = targets[fit_mask.to_numpy()]
    x_calibration = flatten_windows(sequences[calibration_mask.to_numpy()])
    y_calibration = targets[calibration_mask.to_numpy()]
    x_test = flatten_windows(sequences[test_mask.to_numpy()])
    y_test = targets[test_mask.to_numpy()]
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(y_train).astype(np.float32)
    x_calibration_scaled = x_scaler.transform(x_calibration).astype(np.float32)
    y_calibration_scaled = y_scaler.transform(y_calibration).astype(np.float32)
    x_test_scaled = x_scaler.transform(x_test).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test).astype(np.float32)

    model = TabularMlpRegressor(
        input_dim=x_train_scaled.shape[1],
        output_dim=y_train_scaled.shape[1],
    )
    device = torch_device()
    model.to(device)
    print(f"Training MLP fold {fold} on {device}.")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_function = nn.MSELoss()
    dataset = TensorDataset(
        torch.tensor(x_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    calibration_x = torch.tensor(
        x_calibration_scaled,
        dtype=torch.float32,
        device=device,
    )
    calibration_y = torch.tensor(
        y_calibration_scaled,
        dtype=torch.float32,
        device=device,
    )
    best_state = None
    best_loss = float("inf")
    patience_left = config.EARLY_STOPPING_PATIENCE
    epochs_trained = 0
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        epochs_trained += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(calibration_x), calibration_y))
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = config.EARLY_STOPPING_PATIENCE
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
        model.train()
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        calibration_prediction = model(
            torch.tensor(
                x_calibration_scaled,
                dtype=torch.float32,
                device=device,
            )
        ).detach().cpu().numpy()
        test_prediction = model(
            torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
        ).detach().cpu().numpy()
    calibration_scores = anomaly_scores(
        calibration_prediction,
        y_calibration_scaled,
    )
    threshold = calibrate_threshold(calibration_scores)
    test_scores = anomaly_scores(test_prediction, y_test_scaled)
    metrics = evaluate_scores(
        metadata.loc[test_mask, "is_incident"].to_numpy(dtype=int),
        test_scores,
        threshold,
    )
    model_path = model_dir / "model.pt"
    torch.save(
        {
            "state_dict": cpu_state_dict(model),
            "input_dim": x_train_scaled.shape[1],
            "output_dim": y_train_scaled.shape[1],
            "lookback": config.LOOKBACK,
            "feature_columns": config.FEATURE_COLUMNS,
            "threshold": threshold,
            "run_mode": run_mode,
            "threshold_strategy": "baseline_calibration_quantile",
            "threshold_quantile": config.THRESHOLD_QUANTILE,
            "epochs_trained": epochs_trained,
            "device_used": str(device),
        },
        model_path,
    )
    joblib.dump(
        {"x_scaler": x_scaler, "y_scaler": y_scaler},
        model_dir / "scalers.joblib",
    )
    score_path = save_predictions(
        model_dir,
        metadata.loc[test_mask].reset_index(drop=True),
        test_scores,
        threshold,
        model_name,
        fold,
    )
    return TrainingResult(
        model_name=model_name,
        fold=fold,
        threshold=threshold,
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        tp=int(metrics["tp"]),
        fp=int(metrics["fp"]),
        fn=int(metrics["fn"]),
        tn=int(metrics["tn"]),
        train_windows=int(fit_mask.sum()),
        calibration_windows=int(calibration_mask.sum()),
        test_windows=int(test_mask.sum()),
        model_path=str(model_path),
        score_path=str(score_path),
        run_mode=run_mode,
        threshold_strategy="baseline_calibration_quantile",
        epochs_trained=epochs_trained,
    )


def train_lstm_fold(
    sequences: np.ndarray,
    targets: np.ndarray,
    stations: np.ndarray,
    metadata: pd.DataFrame,
    fold: int,
    epochs: int,
    batch_size: int,
    run_mode: str,
    output_root: Path,
) -> TrainingResult:
    """Train the LSTM model for a held-out fold."""

    if torch is None:
        raise ImportError("torch is required for LSTM training.")
    model_name = "lstm"
    model_dir = output_root / model_name / f"fold_{fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_mask, calibration_mask, test_mask = split_fit_calibration_masks(
        metadata,
        fold,
    )
    baseline_mask = fit_mask | calibration_mask
    station_values = sorted(np.unique(stations[baseline_mask.to_numpy()]).tolist())
    station_map = {int(station): index for index, station in enumerate(station_values)}

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_sequences = sequences[fit_mask.to_numpy()]
    calibration_sequences = sequences[calibration_mask.to_numpy()]
    test_sequences = sequences[test_mask.to_numpy()]
    x_scaler.fit(train_sequences.reshape(-1, train_sequences.shape[-1]))
    y_scaler.fit(targets[fit_mask.to_numpy()])
    x_train = x_scaler.transform(
        train_sequences.reshape(-1, train_sequences.shape[-1])
    ).reshape(train_sequences.shape)
    x_calibration = x_scaler.transform(
        calibration_sequences.reshape(-1, calibration_sequences.shape[-1])
    ).reshape(calibration_sequences.shape)
    x_test = x_scaler.transform(
        test_sequences.reshape(-1, test_sequences.shape[-1])
    ).reshape(test_sequences.shape)
    y_train = y_scaler.transform(targets[fit_mask.to_numpy()])
    y_calibration = y_scaler.transform(targets[calibration_mask.to_numpy()])
    y_test = y_scaler.transform(targets[test_mask.to_numpy()])
    train_station = np.array(
        [station_map[int(station)] for station in stations[fit_mask.to_numpy()]],
        dtype=np.int64,
    )
    calibration_station = np.array(
        [
            station_map[int(station)]
            for station in stations[calibration_mask.to_numpy()]
        ],
        dtype=np.int64,
    )
    test_station = np.array(
        [
            station_map.get(int(station), 0)
            for station in stations[test_mask.to_numpy()]
        ],
        dtype=np.int64,
    )

    model = StationLstmRegressor(
        n_stations=len(station_map),
        n_features=len(config.FEATURE_COLUMNS),
    )
    device = torch_device()
    model.to(device)
    print(f"Training LSTM fold {fold} on {device}.")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_function = nn.MSELoss()
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(train_station, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    calibration_x = torch.tensor(
        x_calibration,
        dtype=torch.float32,
        device=device,
    )
    calibration_station_tensor = torch.tensor(
        calibration_station,
        dtype=torch.long,
        device=device,
    )
    calibration_y = torch.tensor(
        y_calibration,
        dtype=torch.float32,
        device=device,
    )
    best_state = None
    best_loss = float("inf")
    patience_left = config.EARLY_STOPPING_PATIENCE
    epochs_trained = 0
    model.train()
    for _ in range(epochs):
        for batch_x, batch_station, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_station = batch_station.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(batch_x, batch_station), batch_y)
            loss.backward()
            optimizer.step()
        epochs_trained += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(
                    model(calibration_x, calibration_station_tensor),
                    calibration_y,
                )
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = config.EARLY_STOPPING_PATIENCE
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
        model.train()
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        calibration_prediction = model(
            torch.tensor(x_calibration, dtype=torch.float32, device=device),
            torch.tensor(calibration_station, dtype=torch.long, device=device),
        ).detach().cpu().numpy()
        test_prediction = model(
            torch.tensor(x_test, dtype=torch.float32, device=device),
            torch.tensor(test_station, dtype=torch.long, device=device),
        ).detach().cpu().numpy()
    calibration_scores = anomaly_scores(calibration_prediction, y_calibration)
    threshold = calibrate_threshold(calibration_scores)
    test_scores = anomaly_scores(test_prediction, y_test)
    metrics = evaluate_scores(
        metadata.loc[test_mask, "is_incident"].to_numpy(dtype=int),
        test_scores,
        threshold,
    )
    model_path = model_dir / "model.pt"
    torch.save(
        {
            "state_dict": cpu_state_dict(model),
            "station_map": station_map,
            "lookback": config.LOOKBACK,
            "feature_columns": config.FEATURE_COLUMNS,
            "threshold": threshold,
            "n_stations": len(station_map),
            "n_features": len(config.FEATURE_COLUMNS),
            "run_mode": run_mode,
            "threshold_strategy": "baseline_calibration_quantile",
            "threshold_quantile": config.THRESHOLD_QUANTILE,
            "epochs_trained": epochs_trained,
            "device_used": str(device),
        },
        model_path,
    )
    joblib.dump(
        {"x_scaler": x_scaler, "y_scaler": y_scaler},
        model_dir / "scalers.joblib",
    )
    score_path = save_predictions(
        model_dir,
        metadata.loc[test_mask].reset_index(drop=True),
        test_scores,
        threshold,
        model_name,
        fold,
    )
    return TrainingResult(
        model_name=model_name,
        fold=fold,
        threshold=threshold,
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        tp=int(metrics["tp"]),
        fp=int(metrics["fp"]),
        fn=int(metrics["fn"]),
        tn=int(metrics["tn"]),
        train_windows=int(fit_mask.sum()),
        calibration_windows=int(calibration_mask.sum()),
        test_windows=int(test_mask.sum()),
        model_path=str(model_path),
        score_path=str(score_path),
        run_mode=run_mode,
        threshold_strategy="baseline_calibration_quantile",
        epochs_trained=epochs_trained,
    )


def train_all(
    max_streams: int | None = None,
    folds: list[int] | None = None,
    epochs: int | None = None,
    batch_size: int = config.BATCH_SIZE,
    run_mode: str = config.DEBUG_RUN_MODE,
) -> list[TrainingResult]:
    """Train all requested model families and persist checkpoints."""

    if run_mode not in config.RUN_MODES:
        raise ValueError(f"run_mode must be one of {config.RUN_MODES}.")
    if epochs is None:
        epochs = (
            config.FINAL_MAX_EPOCHS
            if run_mode == config.FINAL_RUN_MODE
            else config.DEBUG_EPOCHS
        )
    output_root = (
        config.MODELS_DIR / config.FINAL_RUN_MODE
        if run_mode == config.FINAL_RUN_MODE
        else config.MODELS_DIR
    )
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(max_streams=max_streams)
    sequences, targets, stations, metadata = make_windows(dataset)
    selected_folds = folds or sorted(metadata["fold"].unique().tolist())
    factories: dict[str, Callable[[], object]] = {
        "random_forest": lambda: RandomForestRegressor(**config.RF_PARAMS),
    }
    if XGBRegressor is not None:
        factories["xgboost"] = lambda: XGBRegressor(**config.XGB_PARAMS)
    else:
        raise ImportError("xgboost is required for the XGBoost model family.")

    results: list[TrainingResult] = []
    for fold in selected_folds:
        for model_name, factory in factories.items():
            results.append(
                train_tabular_fold(
                    model_name,
                    factory,
                    sequences,
                    targets,
                    metadata,
                    int(fold),
                    run_mode=run_mode,
                    output_root=output_root,
                )
            )
        results.append(
            train_mlp_fold(
                sequences,
                targets,
                metadata,
                int(fold),
                epochs=epochs,
                batch_size=batch_size,
                run_mode=run_mode,
                output_root=output_root,
            )
        )
        results.append(
            train_lstm_fold(
                sequences,
                targets,
                stations,
                metadata,
                int(fold),
                epochs=epochs,
                batch_size=batch_size,
                run_mode=run_mode,
                output_root=output_root,
            )
        )
    manifest_path = (
        config.FINAL_MANIFEST_PATH
        if run_mode == config.FINAL_RUN_MODE
        else config.DEBUG_MANIFEST_PATH
    )
    existing_results: list[dict[str, object]] = []
    if manifest_path.exists():
        existing_results = json.loads(manifest_path.read_text(encoding="utf-8"))
    merged: dict[tuple[str, int], dict[str, object]] = {
        (str(result["model_name"]), int(result["fold"])): result
        for result in existing_results
    }
    for result in results:
        merged[(result.model_name, result.fold)] = asdict(result)
    manifest_path.write_text(
        json.dumps(list(merged.values()), indent=2),
        encoding="utf-8",
    )
    if run_mode == config.FINAL_RUN_MODE:
        write_final_training_tables(list(merged.values()))
    return results


def write_final_training_tables(records: list[dict[str, object]]) -> None:
    """Write report-ready final training metric and threshold tables."""

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if "run_mode" in frame.columns:
        final_frame = frame[frame["run_mode"] == config.FINAL_RUN_MODE].copy()
    else:
        final_frame = pd.DataFrame()
    if final_frame.empty:
        final_frame = frame.copy()
    metric_columns = [
        "model_name",
        "fold",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "tp",
        "fp",
        "fn",
        "tn",
        "threshold",
        "threshold_strategy",
        "train_windows",
        "calibration_windows",
        "test_windows",
        "epochs_trained",
        "run_mode",
        "model_path",
        "score_path",
    ]
    available_metrics = [
        column for column in metric_columns if column in final_frame.columns
    ]
    final_frame.loc[:, available_metrics].to_csv(
        config.FINAL_MODEL_METRICS_PATH,
        index=False,
    )
    final_frame.loc[
        :,
        [
            "model_name",
            "fold",
            "threshold",
            "threshold_strategy",
            "run_mode",
        ],
    ].to_csv(config.FINAL_THRESHOLDS_PATH, index=False)
    comparison = (
        final_frame.groupby("model_name")
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            false_positives_total=("fp", "sum"),
            missed_incidents_total=("fn", "sum"),
            true_positives_total=("tp", "sum"),
            true_negatives_total=("tn", "sum"),
        )
        .reset_index()
    )
    comparison["detection_rate_mean"] = comparison["recall_mean"]
    comparison.to_csv(config.FINAL_MODEL_COMPARISON_PATH, index=False)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-streams", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--run-mode",
        choices=config.RUN_MODES,
        default=config.DEBUG_RUN_MODE,
    )
    args = parser.parse_args()
    selected_epochs = args.epochs
    if selected_epochs is None:
        selected_epochs = (
            config.FINAL_MAX_EPOCHS
            if args.run_mode == config.FINAL_RUN_MODE
            else config.DEBUG_EPOCHS
        )
    print(
        "Training command resolved with "
        f"run_mode={args.run_mode}, epochs={selected_epochs}, "
        f"batch_size={args.batch_size}, folds={args.folds or 'all'}."
    )
    if args.run_mode == config.FINAL_RUN_MODE:
        print(f"Expected manifest: {config.FINAL_MANIFEST_PATH}")
        print(f"Expected metrics: {config.FINAL_MODEL_METRICS_PATH}")
        print(f"Expected thresholds: {config.FINAL_THRESHOLDS_PATH}")
        print(f"Expected comparison: {config.FINAL_MODEL_COMPARISON_PATH}")
    results = train_all(
        max_streams=args.max_streams,
        folds=args.folds,
        epochs=selected_epochs,
        batch_size=args.batch_size,
        run_mode=args.run_mode,
    )
    print(f"Saved {len(results)} trained fold artifacts under {config.MODELS_DIR}.")


if __name__ == "__main__":
    main()
