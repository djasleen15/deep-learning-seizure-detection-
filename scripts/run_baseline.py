"""Run the amplitude-threshold baseline on one CHB-MIT EDF file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from seizure_detection.baseline import (
    baseline_multichannel_amplitude_detector,
    compute_binary_metrics_clean,
    compute_nonseizure_amplitude_stats,
)
from seizure_detection.data_processing import (
    create_windows,
    load_all_annotations,
    load_edf,
    preprocess_signal,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to chb-mit-data folder")
    parser.add_argument("--patient", default="chb01")
    parser.add_argument("--filename", default="chb01_04.edf")
    parser.add_argument("--output-dir", default="report_figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_all_annotations(data_dir, [args.patient])
    edf_data = load_edf(data_dir / args.patient / args.filename)
    processed = preprocess_signal(edf_data["signals"], edf_data["fs"])
    seizure_intervals = annotations[args.patient][args.filename]["seizures"]

    windows, labels, _ = create_windows(
        processed,
        edf_data["fs"],
        seizure_intervals,
    )
    mean_amp, std_amp = compute_nonseizure_amplitude_stats(windows, labels)

    grid_results = []
    for threshold_std in [1.5, 2, 2.5, 3]:
        for min_channels in [1, 2, 3]:
            preds, threshold = baseline_multichannel_amplitude_detector(
                windows,
                mean_amp,
                std_amp,
                edf_data["fs"],
                threshold_std=threshold_std,
                min_duration_sec=2,
                min_channels=min_channels,
            )
            metrics = compute_binary_metrics_clean(labels, preds)
            grid_results.append(
                {
                    "threshold_rule": f"mean + {threshold_std}*std",
                    "threshold_value": round(float(threshold), 2),
                    "min_channels": min_channels,
                    "predicted_seizure_windows": int(np.sum(preds == 1)),
                    **metrics,
                }
            )

    grid_df = pd.DataFrame(grid_results).sort_values(
        by=["f1", "sensitivity_recall", "specificity"],
        ascending=False,
    )
    grid_df.to_csv(output_dir / "baseline_threshold_grid.csv", index=False)
    print(grid_df)


if __name__ == "__main__":
    main()

