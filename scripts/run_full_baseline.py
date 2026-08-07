"""Run the fixed amplitude-threshold baseline on train/validation patients.

The threshold is computed once from training-patient non-seizure windows only,
then applied unchanged to both train and validation splits.

This keeps the baseline aligned with strict patient-level generalization:
validation patients are not used to calibrate the threshold.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seizure_detection.baseline import (
    baseline_multichannel_amplitude_detector,
    compute_binary_metrics_clean,
)
from seizure_detection.data_processing import (
    create_windows,
    load_all_annotations,
    load_edf,
    preprocess_signal,
)
from seizure_detection.splits import TRAIN_PATIENTS, VAL_PATIENTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to chb-mit-data folder")
    parser.add_argument("--output-dir", required=True, help="Where baseline outputs are saved")
    parser.add_argument("--threshold-std", type=float, default=2.5)
    parser.add_argument("--min-channels", type=int, default=2)
    parser.add_argument("--min-duration-sec", type=float, default=2.0)
    parser.add_argument(
        "--include-all-files",
        action="store_true",
        help="Evaluate all EDF files, not only seizure-containing EDF files.",
    )
    return parser.parse_args()


def patient_edf_files(data_dir, patient):
    patient_dir = data_dir / patient
    return sorted(path.name for path in patient_dir.glob("*.edf"))


def selected_files_for_patient(data_dir, annotations, patient, include_all_files):
    available_edfs = patient_edf_files(data_dir, patient)

    if include_all_files:
        return available_edfs

    return [
        filename
        for filename in available_edfs
        if annotations[patient].get(filename, {}).get("num_seizures", 0) > 0
    ]


def load_preprocessed_windows(data_dir, annotations, patient, filename):
    edf_data = load_edf(data_dir / patient / filename)
    processed = preprocess_signal(edf_data["signals"], edf_data["fs"])
    seizure_intervals = annotations[patient][filename]["seizures"]

    windows, labels, window_times = create_windows(
        processed,
        edf_data["fs"],
        seizure_intervals,
    )

    return windows, labels, window_times, edf_data["fs"]


def compute_global_train_nonseizure_stats(data_dir, annotations, patients, include_all_files):
    total_count = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    files_seen = 0

    for patient in patients:
        filenames = selected_files_for_patient(
            data_dir,
            annotations,
            patient,
            include_all_files,
        )

        print(f"\nStats pass {patient}: {len(filenames)} files")

        for filename in filenames:
            print(f"  {patient}/{filename}")
            windows, labels, _, _ = load_preprocessed_windows(
                data_dir,
                annotations,
                patient,
                filename,
            )

            nonseizure_windows = windows[labels == 0]
            abs_values = np.abs(nonseizure_windows)

            total_count += abs_values.size
            total_sum += float(np.sum(abs_values))
            total_sum_sq += float(np.sum(abs_values ** 2))
            files_seen += 1

    mean_amp = total_sum / total_count
    variance = (total_sum_sq / total_count) - (mean_amp ** 2)
    std_amp = float(np.sqrt(max(variance, 0.0)))

    return {
        "mean_amp": float(mean_amp),
        "std_amp": std_amp,
        "files_seen": files_seen,
        "values_seen": int(total_count),
    }


def evaluate_split(
    data_dir,
    annotations,
    split_name,
    patients,
    mean_amp,
    std_amp,
    threshold_std,
    min_channels,
    min_duration_sec,
    include_all_files,
):
    all_true = []
    all_pred = []
    file_rows = []

    first_tp = None
    first_fp = None

    for patient in patients:
        filenames = selected_files_for_patient(
            data_dir,
            annotations,
            patient,
            include_all_files,
        )

        print(f"\nEval {split_name} {patient}: {len(filenames)} files")

        for filename in filenames:
            print(f"  {patient}/{filename}")

            windows, labels, window_times, fs = load_preprocessed_windows(
                data_dir,
                annotations,
                patient,
                filename,
            )

            preds, threshold = baseline_multichannel_amplitude_detector(
                windows,
                mean_amp,
                std_amp,
                fs,
                threshold_std=threshold_std,
                min_duration_sec=min_duration_sec,
                min_channels=min_channels,
            )

            metrics = compute_binary_metrics_clean(labels, preds)

            file_rows.append(
                {
                    "split": split_name,
                    "patient": patient,
                    "filename": filename,
                    "num_windows": len(labels),
                    "num_seizure_windows": int(np.sum(labels == 1)),
                    "num_nonseizure_windows": int(np.sum(labels == 0)),
                    **metrics,
                }
            )

            all_true.extend(labels)
            all_pred.extend(preds)

            if split_name == "val":
                tp_indices = np.where((labels == 1) & (preds == 1))[0]
                fp_indices = np.where((labels == 0) & (preds == 1))[0]

                if first_tp is None and len(tp_indices) > 0:
                    idx = int(tp_indices[0])
                    first_tp = {
                        "patient": patient,
                        "filename": filename,
                        "window_idx": idx,
                        "window": windows[idx],
                        "time": window_times[idx],
                        "fs": fs,
                    }

                if first_fp is None and len(fp_indices) > 0:
                    idx = int(fp_indices[0])
                    first_fp = {
                        "patient": patient,
                        "filename": filename,
                        "window_idx": idx,
                        "window": windows[idx],
                        "time": window_times[idx],
                        "fs": fs,
                    }

    split_metrics = compute_binary_metrics_clean(
        np.array(all_true),
        np.array(all_pred),
    )

    split_metrics = {
        "split": split_name,
        "num_patients": len(patients),
        "num_files": len(file_rows),
        "num_windows": len(all_true),
        **split_metrics,
    }

    return split_metrics, file_rows, first_tp, first_fp


def max_amplitude_channel(window):
    return int(np.argmax(np.max(np.abs(window), axis=1)))


def save_qualitative_plot(tp_example, fp_example, threshold, save_path):
    if tp_example is None or fp_example is None:
        print("WARNING: could not save qualitative plot because TP or FP example is missing.")
        return

    examples = [
        ("True positive seizure window", tp_example),
        ("False positive non-seizure window", fp_example),
    ]

    plt.figure(figsize=(12, 5))

    for plot_idx, (title, example) in enumerate(examples, start=1):
        window = example["window"]
        channel_idx = max_amplitude_channel(window)
        time_axis = np.arange(window.shape[1]) / example["fs"]
        abs_signal = np.abs(window[channel_idx])

        plt.subplot(1, 2, plot_idx)
        plt.plot(time_axis, abs_signal, linewidth=1)
        plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
        plt.title(
            f"{title}\n"
            f"{example['patient']}/{example['filename']}, "
            f"window {example['window_idx']}"
        )
        plt.xlabel("Time within 4-second window (s)")
        plt.ylabel("Absolute amplitude")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved qualitative plot:", save_path)


def write_dicts_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patients = TRAIN_PATIENTS + VAL_PATIENTS
    annotations = load_all_annotations(data_dir, patients)

    print("Computing global train-only non-seizure amplitude statistics...")
    stats = compute_global_train_nonseizure_stats(
        data_dir,
        annotations,
        TRAIN_PATIENTS,
        include_all_files=args.include_all_files,
    )

    threshold = stats["mean_amp"] + args.threshold_std * stats["std_amp"]

    print("\nBaseline threshold stats")
    print(stats)
    print("threshold_std:", args.threshold_std)
    print("min_channels:", args.min_channels)
    print("min_duration_sec:", args.min_duration_sec)
    print("threshold:", threshold)

    train_metrics, train_file_rows, _, _ = evaluate_split(
        data_dir,
        annotations,
        "train",
        TRAIN_PATIENTS,
        stats["mean_amp"],
        stats["std_amp"],
        args.threshold_std,
        args.min_channels,
        args.min_duration_sec,
        args.include_all_files,
    )

    val_metrics, val_file_rows, tp_example, fp_example = evaluate_split(
        data_dir,
        annotations,
        "val",
        VAL_PATIENTS,
        stats["mean_amp"],
        stats["std_amp"],
        args.threshold_std,
        args.min_channels,
        args.min_duration_sec,
        args.include_all_files,
    )

    metrics_rows = [
        {
            "split": "train",
            "mean_amp": stats["mean_amp"],
            "std_amp": stats["std_amp"],
            "threshold": threshold,
            "threshold_std": args.threshold_std,
            "min_channels": args.min_channels,
            "min_duration_sec": args.min_duration_sec,
            **train_metrics,
        },
        {
            "split": "val",
            "mean_amp": stats["mean_amp"],
            "std_amp": stats["std_amp"],
            "threshold": threshold,
            "threshold_std": args.threshold_std,
            "min_channels": args.min_channels,
            "min_duration_sec": args.min_duration_sec,
            **val_metrics,
        },
    ]

    metrics_path = output_dir / "baseline_train_val_metrics.csv"
    file_metrics_path = output_dir / "baseline_file_metrics.csv"
    qualitative_path = output_dir / "baseline_val_true_positive_vs_false_positive.png"

    write_dicts_csv(metrics_path, metrics_rows)
    write_dicts_csv(file_metrics_path, train_file_rows + val_file_rows)
    save_qualitative_plot(tp_example, fp_example, threshold, qualitative_path)

    print("\nFinal baseline metrics")
    for row in metrics_rows:
        print(row)

    print("Saved metrics:", metrics_path)
    print("Saved file-level metrics:", file_metrics_path)


if __name__ == "__main__":
    main()
