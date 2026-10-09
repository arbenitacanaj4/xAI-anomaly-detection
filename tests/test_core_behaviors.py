import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_incident_xai import config
from traffic_incident_xai.data_pipeline import assign_folds, synchronized_windows
from traffic_incident_xai.model_training import (
    anomaly_scores,
    calibrate_threshold,
    evaluate_scores,
    make_windows,
    save_predictions,
    split_fit_calibration_masks,
)
from traffic_incident_xai.triage import build_alarm_episodes


def synthetic_rows(stream_id, split, start_value, fold=1, count=4, station_id=10):
    timestamps = pd.date_range("2024-01-01", periods=count, freq="5min")
    incident_values = ([0, 0, 1, 1] + [0] * count)[:count]
    return pd.DataFrame(
        {
            "row_id": np.arange(start_value, start_value + count),
            "stream_id": stream_id,
            "source_file": f"incident_{stream_id}.csv",
            "timestamp": timestamps,
            "station_id": station_id,
            "row_in_split": np.arange(count),
            "split": split,
            "fold": fold,
            "speed_smoothed": np.arange(start_value, start_value + count, dtype=float),
            "occ_smoothed": np.arange(
                start_value + 100,
                start_value + 100 + count,
                dtype=float,
            ),
            "is_incident": incident_values,
        }
    )


class CoreBehaviorTests(unittest.TestCase):
    def patch_config(self, name, value):
        original = getattr(config, name)
        setattr(config, name, value)
        self.addCleanup(setattr, config, name, original)

    def test_synchronized_windows_uses_strict_train_test_boundaries(self):
        self.patch_config("TRAIN_ROWS", 5)
        self.patch_config("TEST_ROWS", 3)
        frame = synthetic_rows(9, "raw", 0, count=9)

        train, test = synchronized_windows(frame)

        self.assertEqual(train["row_id"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(test["row_id"].tolist(), [5, 6, 7])
        self.assertEqual(train["timestamp"].iloc[-1], frame["timestamp"].iloc[4])
        self.assertEqual(test["timestamp"].iloc[0], frame["timestamp"].iloc[5])

        train.loc[0, "speed_smoothed"] = -999.0
        self.assertNotEqual(frame.loc[0, "speed_smoothed"], -999.0)

    def test_synchronized_windows_rejects_short_streams(self):
        self.patch_config("TRAIN_ROWS", 5)
        self.patch_config("TEST_ROWS", 3)
        frame = synthetic_rows(9, "raw", 0, count=7)

        with self.assertRaises(ValueError):
            synchronized_windows(frame)

    def test_make_windows_stays_within_stream_and_split_boundaries(self):
        frame = pd.concat(
            [
                synthetic_rows(1, "train", 0, fold=1),
                synthetic_rows(1, "test", 100, fold=1),
                synthetic_rows(2, "train", 200, fold=2),
                synthetic_rows(2, "test", 300, fold=2),
            ],
            ignore_index=True,
        )

        sequences, targets, stations, metadata = make_windows(frame, lookback=2)

        self.assertEqual(sequences.shape, (8, 2, 2))
        self.assertEqual(targets.shape, (8, 2))
        self.assertEqual(stations.tolist(), [10] * 8)
        self.assertTrue((metadata["row_in_split"] >= 2).all())

        first_train = metadata[
            (metadata["stream_id"] == 1)
            & (metadata["split"] == "train")
            & (metadata["row_in_split"] == 2)
        ].index[0]
        np.testing.assert_array_equal(sequences[first_train, :, 0], [0.0, 1.0])
        np.testing.assert_array_equal(targets[first_train], [2.0, 102.0])

        first_test = metadata[
            (metadata["stream_id"] == 1)
            & (metadata["split"] == "test")
            & (metadata["row_in_split"] == 2)
        ].index[0]
        np.testing.assert_array_equal(sequences[first_test, :, 0], [100.0, 101.0])
        np.testing.assert_array_equal(targets[first_test], [102.0, 202.0])

    def test_make_windows_rejects_missing_feature_columns(self):
        frame = synthetic_rows(1, "train", 0).drop(columns=["occ_smoothed"])

        with self.assertRaises(KeyError):
            make_windows(frame, lookback=2)

    def test_assign_folds_is_deterministic_and_balanced(self):
        self.patch_config("N_FOLDS", 3)
        stream_ids = [16, 11, 12, 10, 15, 13, 14]

        first = assign_folds(stream_ids)
        second = assign_folds(list(reversed(stream_ids)))

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(stream_ids))
        self.assertEqual(set(first.values()), {1, 2, 3})
        fold_sizes = pd.Series(first).value_counts().sort_values().tolist()
        self.assertEqual(fold_sizes, [2, 2, 3])

    def test_split_fit_calibration_masks_exclude_heldout_fold(self):
        self.patch_config("TRAIN_ROWS", 10)
        self.patch_config("CALIBRATION_FRACTION", 0.20)
        metadata = pd.concat(
            [
                pd.DataFrame(
                    {
                        "split": "train",
                        "fold": 1,
                        "row_in_split": np.arange(10),
                    }
                ),
                pd.DataFrame(
                    {
                        "split": "train",
                        "fold": 2,
                        "row_in_split": np.arange(10),
                    }
                ),
                pd.DataFrame(
                    {
                        "split": "test",
                        "fold": 2,
                        "row_in_split": np.arange(5),
                    }
                ),
            ],
            ignore_index=True,
        )

        fit, calibration, heldout = split_fit_calibration_masks(metadata, fold=2)

        self.assertEqual(int(fit.sum()), 8)
        self.assertEqual(int(calibration.sum()), 2)
        self.assertEqual(int(heldout.sum()), 5)
        self.assertTrue((metadata.loc[fit | calibration, "fold"] == 1).all())
        self.assertTrue((metadata.loc[heldout, "split"] == "test").all())

    def test_anomaly_scores_use_rowwise_root_mean_squared_residuals(self):
        predicted = np.array([[1.0, 2.0], [4.0, 6.0]])
        actual = np.array([[1.0, 0.0], [1.0, 2.0]])

        scores = anomaly_scores(predicted, actual)

        np.testing.assert_allclose(scores, [np.sqrt(2.0), np.sqrt(12.5)])
        np.testing.assert_array_equal(anomaly_scores(actual, actual), [0.0, 0.0])

    def test_anomaly_scores_reject_one_dimensional_inputs(self):
        with self.assertRaises(Exception):
            anomaly_scores(np.array([1.0, 2.0]), np.array([1.0, 1.0]))

    def test_calibrate_threshold_uses_configured_quantile(self):
        self.patch_config("THRESHOLD_QUANTILE", 0.75)
        scores = np.array([0.0, 1.0, 2.0, 3.0])

        self.assertEqual(calibrate_threshold(scores), 2.25)

    def test_calibrate_threshold_rejects_empty_scores(self):
        with self.assertRaises(Exception):
            calibrate_threshold(np.array([]))

    def test_evaluate_scores_uses_strict_threshold_and_handles_single_class_labels(self):
        metrics = evaluate_scores(
            labels=np.array([0, 1, 1, 0]),
            scores=np.array([0.1, 0.5, 0.6, 0.8]),
            threshold=0.5,
        )

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

        single_class = evaluate_scores(
            labels=np.array([0, 0]),
            scores=np.array([0.2, 0.7]),
            threshold=0.5,
        )
        self.assertIsNone(single_class["roc_auc"])
        self.assertIsNone(single_class["pr_auc"])

    def test_save_predictions_writes_to_temp_dir_and_derives_alarm_columns(self):
        metadata = synthetic_rows(1, "test", 0, count=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = save_predictions(
                Path(temp_dir),
                metadata,
                scores=np.array([0.5, 0.6]),
                threshold=0.5,
                model_name="synthetic_model",
                fold=1,
            )

            saved = pd.read_csv(output_path)

        self.assertTrue(str(output_path).endswith("synthetic_model_fold_1_scores.csv"))
        self.assertEqual(saved["prediction"].tolist(), [0, 1])
        self.assertEqual(
            saved["case_type"].tolist(),
            ["true_negative", "false_positive"],
        )
        np.testing.assert_allclose(saved["threshold_margin"], [0.0, 0.1])

    def test_build_alarm_episodes_groups_consecutive_positives_per_stream(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            xai_path = Path(temp_dir) / "xai.csv"
            agreement_path = Path(temp_dir) / "agreement.csv"
            pd.DataFrame(
                {
                    "case_id": ["a", "b", "c", "d"],
                    "row_id": [2, 4, 6, 101],
                    "model": ["m", "m", "m", "m"],
                    "feature": ["speed_lag_1", "occ_lag_1", "speed_lag_2", "occ_lag_2"],
                    "abs_attribution": [0.4, 0.2, 0.1, 0.3],
                }
            ).to_csv(xai_path, index=False)
            pd.DataFrame(
                {
                    "case_id": ["a", "b", "c", "d"],
                    "model": ["m", "m", "m", "m"],
                    "explanation_agreement_level": [
                        "consistent",
                        "consistent",
                        "conflicting",
                        "consistent",
                    ],
                }
            ).to_csv(agreement_path, index=False)
            self.patch_config(
                "XAI_LOCAL_EXPLANATIONS_PATH",
                xai_path,
            )
            self.patch_config(
                "XAI_METHOD_AGREEMENT_PATH",
                agreement_path,
            )
            times = pd.date_range("2024-01-01", periods=7, freq="5min")
            stream_one = pd.DataFrame(
                {
                    "model": "m",
                    "fold": 1,
                    "stream_id": 10,
                    "source_file": "incident_10.csv",
                    "row_id": np.arange(7),
                    "timestamp": times,
                    "prediction": [0, 1, 1, 0, 1, 0, 1],
                    "true_label": [0, 0, 1, 0, 0, 0, 0],
                    "normalized_score": [0.1, 0.7, 0.8, 0.2, 0.65, 0.1, 0.9],
                    "threshold_margin": [-0.1, 0.2, 0.3, -0.2, 0.1, -0.1, 0.4],
                    "speed_smoothed": [60, 58, 55, 59, 57, 60, 54],
                    "occ_smoothed": [10, 12, 15, 11, 13, 10, 16],
                }
            )
            stream_two = stream_one.copy()
            stream_two["stream_id"] = 20
            stream_two["source_file"] = "incident_20.csv"
            stream_two["row_id"] = np.arange(100, 107)
            stream_two["prediction"] = [1, 1, 0, 0, 0, 0, 0]
            stream_two["true_label"] = 0

            episodes = build_alarm_episodes(
                pd.concat([stream_one, stream_two], ignore_index=True)
            )

        stream_one_episodes = episodes[episodes["stream_id"] == 10]
        stream_two_episodes = episodes[episodes["stream_id"] == 20]
        self.assertEqual(stream_one_episodes["alarm_duration_timestamps"].tolist(), [2, 1, 1])
        self.assertEqual(stream_two_episodes["alarm_duration_timestamps"].tolist(), [2])
        self.assertEqual(int(stream_one_episodes.iloc[0]["representative_row_id"]), 2)
        self.assertEqual(
            stream_one_episodes.iloc[0]["evaluation_episode_type"],
            "true_incident_alarm",
        )
        self.assertEqual(
            stream_one_episodes.iloc[1]["evaluation_episode_type"],
            "false_alarm_episode",
        )

    def test_build_alarm_episodes_handles_xai_without_method_agreement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            xai_path = Path(temp_dir) / "xai.csv"
            pd.DataFrame(
                {
                    "case_id": ["a"],
                    "row_id": [1],
                    "model": ["m"],
                    "feature": ["speed_lag_1"],
                    "abs_attribution": [0.4],
                }
            ).to_csv(xai_path, index=False)
            self.patch_config("XAI_LOCAL_EXPLANATIONS_PATH", xai_path)
            self.patch_config(
                "XAI_METHOD_AGREEMENT_PATH",
                Path(temp_dir) / "missing_agreement.csv",
            )
            predictions = synthetic_rows(1, "test", 0, count=3).assign(
                model="m",
                true_label=[0, 1, 0],
                prediction=[0, 1, 0],
                normalized_score=[0.1, 0.8, 0.1],
                threshold_margin=[-0.2, 0.3, -0.2],
            )

            episodes = build_alarm_episodes(predictions)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(
            episodes.iloc[0]["explanation_agreement_level"],
            "not_available",
        )

    def test_build_alarm_episodes_handles_missing_optional_xai_support(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.patch_config(
                "XAI_LOCAL_EXPLANATIONS_PATH",
                Path(temp_dir) / "missing_xai.csv",
            )
            predictions = synthetic_rows(1, "test", 0, count=3).assign(
                model="m",
                true_label=[0, 1, 0],
                prediction=[0, 1, 0],
                normalized_score=[0.1, 0.8, 0.1],
                threshold_margin=[-0.2, 0.3, -0.2],
            )

            episodes = build_alarm_episodes(predictions)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes.iloc[0]["dominant_xai_features"], "not_available")

    def test_build_alarm_episodes_returns_empty_frame_when_no_predictions_fire(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.patch_config(
                "XAI_LOCAL_EXPLANATIONS_PATH",
                Path(temp_dir) / "missing_xai.csv",
            )
            predictions = synthetic_rows(1, "test", 0, count=3).assign(
                model="m",
                true_label=0,
                prediction=0,
                normalized_score=0.1,
                threshold_margin=-0.2,
            )

            episodes = build_alarm_episodes(predictions)

        self.assertTrue(episodes.empty)


if __name__ == "__main__":
    unittest.main()
