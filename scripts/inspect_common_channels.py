"""Inspect common EEG channel labels across processed train/validation files.

This script does not touch the test split. It uses the per-EDF processed files
from Chunk 1 to determine which raw EDF headers should be inspected, then computes
the channel-label intersection across train + validation files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np

from seizure_detection.data_processing import load_edf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to chb-mit-data folder")
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Path to processed_patient_splits folder",
    )
    parser.add_argument("--output-dir", required=True, help="Where channel reports are saved")
    parser.add_argument(
        "--from-npz-files",
        action="store_true",
        help=(
            "Inspect every .npz in processed train/val split folders instead of "
            "using summary CSVs. Useful when summaries were created in staged runs."
        ),
    )
    parser.add_argument(
        "--min-file-coverage",
        type=float,
        default=1.0,
        help=(
            "Minimum fraction of train/val files that must contain a label. "
            "Use 1.0 for exact intersection; 0.95 allows rare incompatible "
            "CHB-MIT files to be excluded later."
        ),
    )
    parser.add_argument(
        "--ignore-labels",
        nargs="*",
        default=["-", "."],
        help="Channel labels to ignore as placeholders/non-EEG labels.",
    )
    return parser.parse_args()


def read_processed_summary(summary_path):
    rows = []

    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


def read_processed_npz_files(processed_dir):
    rows = []

    for split in ["train", "val"]:
        split_dir = processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing processed split folder: {split_dir}")

        for npz_path in sorted(split_dir.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            rows.append(
                {
                    "split": split,
                    "patient": str(data["patient_id"]),
                    "filename": str(data["file_id"]),
                    "npz_path": str(npz_path),
                }
            )

    return rows


def channel_labels_for_file(data_dir, patient, filename):
    edf_data = load_edf(data_dir / patient / filename)
    return list(edf_data["channel_labels"])


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_npz_files:
        all_rows = read_processed_npz_files(processed_dir)
    else:
        summary_paths = [
            processed_dir / "train_processed_files.csv",
            processed_dir / "val_processed_files.csv",
        ]

        all_rows = []

        for summary_path in summary_paths:
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing processed summary: {summary_path}")

            split = "train" if "train" in summary_path.name else "val"

            for row in read_processed_summary(summary_path):
                row["split"] = split
                all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No processed train/validation rows found.")

    if not (0 < args.min_file_coverage <= 1):
        raise ValueError("--min-file-coverage must be in the interval (0, 1]")

    channel_sets = []
    channel_rows = []

    for row in all_rows:
        patient = row["patient"]
        filename = row["filename"]
        split = row["split"]

        labels = channel_labels_for_file(data_dir, patient, filename)
        valid_labels = [label for label in labels if label not in args.ignore_labels]
        channel_sets.append(set(valid_labels))

        channel_rows.append(
            {
                "split": split,
                "patient": patient,
                "filename": filename,
                "num_channels": len(labels),
                "channel_labels": "|".join(labels),
            }
        )

        print(f"{split} {patient}/{filename}: {len(labels)} channels")

    if args.min_file_coverage == 1.0:
        common_channels = sorted(set.intersection(*channel_sets))
        min_required_files = len(channel_sets)
    else:
        min_required_files = math.ceil(args.min_file_coverage * len(channel_sets))
        label_counts = Counter()

        for labels in channel_sets:
            for label in labels:
                label_counts[label] += 1

        common_channels = sorted(
            label for label, count in label_counts.items() if count >= min_required_files
        )

    excluded_for_selected_channels = []
    for row, labels in zip(channel_rows, channel_sets):
        missing = [label for label in common_channels if label not in labels]
        if missing:
            excluded_for_selected_channels.append(
                {
                    "split": row["split"],
                    "patient": row["patient"],
                    "filename": row["filename"],
                    "missing_selected_channels": missing,
                }
            )

    print("\nCommon channel count:", len(common_channels))
    print("Minimum files required per selected channel:", min_required_files)
    print("Files missing selected channels:", len(excluded_for_selected_channels))
    print("Common channels:")
    for label in common_channels:
        print(" ", label)

    if excluded_for_selected_channels:
        print("\nFiles that will be excluded by selected channels:")
        for row in excluded_for_selected_channels:
            print(
                f"  {row['split']} {row['patient']}/{row['filename']}: "
                f"missing {row['missing_selected_channels']}"
            )

    channel_report_path = output_dir / "train_val_channel_labels.csv"
    common_channels_path = output_dir / "common_train_val_channels.json"

    with channel_report_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "patient", "filename", "num_channels", "channel_labels"],
        )
        writer.writeheader()
        writer.writerows(channel_rows)

    with common_channels_path.open("w") as f:
        json.dump(
            {
                "common_channel_count": len(common_channels),
                "common_channels": common_channels,
                "num_files_inspected": len(channel_rows),
                "splits_inspected": ["train", "val"],
                "test_split_inspected": False,
                "min_file_coverage": args.min_file_coverage,
                "min_required_files": min_required_files,
                "ignored_labels": args.ignore_labels,
                "files_missing_selected_channels": excluded_for_selected_channels,
            },
            f,
            indent=2,
        )

    print("\nSaved:", channel_report_path)
    print("Saved:", common_channels_path)


if __name__ == "__main__":
    main()
