"""Inspect common EEG channel labels across processed train/validation files.

This script does not touch the test split. It uses the per-EDF processed files
from Chunk 1 to determine which raw EDF headers should be inspected, then computes
the channel-label intersection across train + validation files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
    return parser.parse_args()


def read_processed_summary(summary_path):
    rows = []

    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

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

    channel_sets = []
    channel_rows = []

    for row in all_rows:
        patient = row["patient"]
        filename = row["filename"]
        split = row["split"]

        labels = channel_labels_for_file(data_dir, patient, filename)
        channel_sets.append(set(labels))

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

    common_channels = sorted(set.intersection(*channel_sets))

    print("\nCommon channel count:", len(common_channels))
    print("Common channels:")
    for label in common_channels:
        print(" ", label)

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
            },
            f,
            indent=2,
        )

    print("\nSaved:", channel_report_path)
    print("Saved:", common_channels_path)


if __name__ == "__main__":
    main()
