"""Hand-coded baseline models and metrics."""

from __future__ import annotations

import numpy as np


def compute_nonseizure_amplitude_stats(
    windows: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compute mean/std absolute amplitude from non-seizure windows."""
    nonseizure_windows = windows[labels == 0]
    return float(np.mean(np.abs(nonseizure_windows))), float(
        np.std(np.abs(nonseizure_windows))
    )


def baseline_multichannel_amplitude_detector(
    windows: np.ndarray,
    mean_amp: float,
    std_amp: float,
    fs: float,
    threshold_std: float = 2.5,
    min_duration_sec: float = 2,
    min_channels: int = 2,
) -> tuple[np.ndarray, float]:
    """Detect seizure windows using a multi-channel amplitude threshold."""
    threshold = mean_amp + threshold_std * std_amp
    min_samples = int(min_duration_sec * fs)
    preds = []

    for window in windows:
        abs_window = np.abs(window)
        channels_above_threshold = np.sum(abs_window > threshold, axis=0)
        high_multichannel = channels_above_threshold >= min_channels
        preds.append(1 if np.sum(high_multichannel) >= min_samples else 0)

    return np.array(preds), float(threshold)


def compute_binary_metrics_clean(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute binary classification metrics with report-friendly scalar values."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0
    )
    accuracy = (tp + tn) / len(y_true)

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "sensitivity_recall": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
    }

