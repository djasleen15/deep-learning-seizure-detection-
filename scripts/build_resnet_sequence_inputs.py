"""Build ResNet-18-ready CNN-LSTM input sequences from processed EDF spectrograms.

Inputs:
- per-EDF .npz files from processed_patient_splits/{train,val}
- common channel list from model_input_outputs/common_train_val_channels.json
- raw EDF headers for channel-label order

Outputs:
- one .npz per EDF containing:
  X_seq: sequences x 3 windows x 3 RGB channels x 224 x 224
  y_seq: sequence labels
  source_window_indices
  patient_id
  file_id

This script does not touch the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from seizure_detection.data_processing import load_edf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to chb-mit-data folder")
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Path to processed_patient_splits folder",
    )
    parser.add_argument(
        "--common-channels-json",
        required=True,
        help="Path to common_train_val_channels.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where model-input sequence .npz files are saved",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Splits to build. Test intentionally unsupported here.",
    )
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild files even if output .npz already exists.",
    )
    return parser.parse_args()


def load_common_channels(path):
    with Path(path).open("r") as f:
        payload = json.load(f)

    return payload["common_channels"]


def parse_processed_filename(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    patient = str(data["patient_id"])
    filename = str(data["file_id"])
    return patient, filename


def channel_indices_for_common(data_dir, patient, filename, common_channels):
    edf_data = load_edf(Path(data_dir) / patient / filename)
    labels = list(edf_data["channel_labels"])
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    missing = [label for label in common_channels if label not in label_to_index]

    if missing:
        return None, missing, labels

    indices = [label_to_index[label] for label in common_channels]
    return indices, [], labels


def resize_to_resnet_rgb(windows, image_size):
    """Resize standardized spectrogram windows to 3x224x224.

    Input shape:
      windows x common_channels x freq_bins x time_bins

    The common channel and frequency dimensions are flattened into one image
    height dimension before resizing. Bilinear interpolation is used to resize
    the 2D spectrogram image to 224x224. The single-channel image is then
    replicated across 3 channels to match ResNet-18's expected RGB input.
    """
    n_windows, n_channels, n_freqs, n_times = windows.shape

    images = windows.reshape(n_windows, 1, n_channels * n_freqs, n_times)
    images = torch.tensor(images, dtype=torch.float32)

    resized = F.interpolate(
        images,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    rgb = resized.repeat(1, 3, 1, 1)

    return rgb.numpy().astype(np.float32)


def build_sequences_for_file(X_windows, y_windows, seq_len):
    X_seq = []
    y_seq = []
    source_window_indices = []

    for start_idx in range(0, len(y_windows) - seq_len + 1):
        end_idx = start_idx + seq_len

        X_seq.append(X_windows[start_idx:end_idx])
        y_seq.append(1 if np.any(y_windows[start_idx:end_idx] == 1) else 0)
        source_window_indices.append(list(range(start_idx, end_idx)))

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.int64),
        np.array(source_window_indices, dtype=np.int64),
    )


def build_one_file(
    data_dir,
    processed_npz_path,
    output_path,
    common_channels,
    seq_len,
    image_size,
):
    patient, filename = parse_processed_filename(processed_npz_path)

    indices, missing, raw_labels = channel_indices_for_common(
        data_dir,
        patient,
        filename,
        common_channels,
    )

    if missing:
        return {
            "status": "excluded_missing_channels",
            "patient": patient,
            "filename": filename,
            "missing_channels": "|".join(missing),
            "raw_channel_count": len(raw_labels),
            "output_path": "",
        }

    data = np.load(processed_npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    X_common = X[:, indices, :, :]
    X_resnet = resize_to_resnet_rgb(X_common, image_size=image_size)

    X_seq, y_seq, source_window_indices = build_sequences_for_file(
        X_resnet,
        y,
        seq_len=seq_len,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X_seq=X_seq,
        y_seq=y_seq,
        source_window_indices=source_window_indices,
        patient_id=np.array(patient),
        file_id=np.array(filename),
        common_channels=np.array(common_channels),
        image_size=np.array(image_size),
        seq_len=np.array(seq_len),
    )

    return {
        "status": "saved",
        "patient": patient,
        "filename": filename,
        "raw_channel_count": int(X.shape[1]),
        "common_channel_count": len(common_channels),
        "num_windows": int(len(y)),
        "num_sequences": int(len(y_seq)),
        "num_seizure_sequences": int(np.sum(y_seq == 1)),
        "num_nonseizure_sequences": int(np.sum(y_seq == 0)),
        "output_path": str(output_path),
    }


def write_summary_csv(path, rows):
    if not rows:
        return

    keys = sorted({key for row in rows for key in row.keys()})

    with path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            values = [str(row.get(key, "")) for key in keys]
            f.write(",".join(values) + "\n")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)

    common_channels = load_common_channels(args.common_channels_json)

    print("Common channel count:", len(common_channels))
    print("Common channels:", common_channels)
    print("Resize method: bilinear interpolation to 224x224, then replicate to 3 channels.")
    print("Sequences are built one EDF at a time, so they cannot cross file boundaries.")

    all_rows = []

    for split in args.splits:
        split_input_dir = processed_dir / split
        split_output_dir = output_dir / split
        split_npz_files = sorted(split_input_dir.glob("*.npz"))

        print(f"\n=== {split} ===")
        print("Input files:", len(split_npz_files))

        for processed_npz_path in split_npz_files:
            output_path = split_output_dir / processed_npz_path.name.replace(
                "_windows.npz",
                "_resnet_sequences.npz",
            )

            if output_path.exists() and not args.overwrite:
                print("SKIP exists:", output_path)
                row = {
                    "status": "skipped_existing",
                    "split": split,
                    "input_path": str(processed_npz_path),
                    "output_path": str(output_path),
                }
                all_rows.append(row)
                continue

            print("Building:", processed_npz_path.name)
            row = build_one_file(
                data_dir,
                processed_npz_path,
                output_path,
                common_channels,
                seq_len=args.seq_len,
                image_size=args.image_size,
            )
            row["split"] = split
            row["input_path"] = str(processed_npz_path)
            all_rows.append(row)
            print(row)

    summary_path = output_dir / "resnet_sequence_build_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary_path, all_rows)

    print("\nSaved summary:", summary_path)


if __name__ == "__main__":
    main()
