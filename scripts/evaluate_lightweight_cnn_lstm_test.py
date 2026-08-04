"""One-shot held-out test evaluation for the lightweight CNN-LSTM.

This script does not train or tune. It loads one preselected checkpoint and
evaluates processed chb22-chb24 test files using the train/validation-selected
channel list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from seizure_detection.baseline import compute_binary_metrics_clean
from seizure_detection.model import SmallCNNLSTM, count_parameters
from seizure_detection.resnet_input import (
    load_channel_labels_csv,
    load_common_channels,
    processed_npz_files,
)
from train_lightweight_cnn_lstm_full import CompactSpectrogramSequenceDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--common-channels-json", required=True)
    parser.add_argument("--channel-labels-csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", default="lightweight_cnn_lstm_test")
    return parser.parse_args()


def evaluate_with_outputs(model, loader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy().astype(int))

    probs = np.array(all_probs)
    preds = np.array(all_preds).astype(int)
    labels = np.array(all_labels).astype(int)
    metrics = compute_binary_metrics_clean(labels, preds)
    return metrics, probs, preds, labels


def sequence_times_for_dataset(dataset: CompactSpectrogramSequenceDataset):
    starts = []
    ends = []
    patients = []
    filenames = []

    for record_idx, start_idx in dataset.sequence_index:
        record = dataset.file_records[record_idx]
        data = np.load(record["path"], allow_pickle=True)
        window_times = data["window_times"]
        end_idx = start_idx + dataset.seq_len - 1
        starts.append(float(window_times[start_idx][0]))
        ends.append(float(window_times[end_idx][1]))
        patients.append(record["patient"])
        filenames.append(record["filename"])

    return np.array(starts), np.array(ends), patients, filenames


def first_seizure_file(dataset: CompactSpectrogramSequenceDataset) -> Path:
    for record in dataset.file_records:
        data = np.load(record["path"], allow_pickle=True)
        if np.any(data["y"] == 1):
            return Path(record["path"])
    raise RuntimeError("No seizure-containing test file found for qualitative plot.")


def plot_probability_trace(
    output_path: Path,
    starts,
    probs,
    labels,
    patient,
    filename,
):
    plt.figure(figsize=(10, 4))
    plt.plot(starts, probs, label="Predicted seizure probability", linewidth=1.5)
    plt.fill_between(
        starts,
        0,
        1,
        where=labels.astype(bool),
        alpha=0.2,
        label="True seizure sequence",
    )
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Threshold")
    plt.xlabel("Time in recording (s)")
    plt.ylabel("Predicted probability")
    plt.title(f"Held-out test prediction trace: {patient} {filename}")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_channels = load_common_channels(args.common_channels_json)
    channel_labels_by_file = (
        load_channel_labels_csv(args.channel_labels_csv)
        if args.channel_labels_csv
        else None
    )
    test_files = processed_npz_files(args.processed_dir, "test")
    if not test_files:
        raise FileNotFoundError(
            f"No processed test files found in {Path(args.processed_dir) / 'test'}"
        )

    dataset = CompactSpectrogramSequenceDataset(
        args.data_dir,
        test_files,
        common_channels,
        channel_labels_by_file,
        seq_len=args.seq_len,
        cache_files=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNNLSTM(feature_dim=128, lstm_hidden=64).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics, probs, preds, labels = evaluate_with_outputs(model, loader, device)
    starts, ends, patients, filenames = sequence_times_for_dataset(dataset)

    probs_path = output_dir / f"{args.run_name}_sequence_predictions.csv"
    pd.DataFrame(
        {
            "patient": patients,
            "filename": filenames,
            "sequence_start_sec": starts,
            "sequence_end_sec": ends,
            "true_label": labels,
            "predicted_label": preds,
            "predicted_probability": probs,
        }
    ).to_csv(probs_path, index=False)

    metrics_payload = {
        "checkpoint": str(args.checkpoint),
        "test_patients": ["chb22", "chb23", "chb24"],
        "common_channel_count": len(common_channels),
        "num_test_files_input": len(test_files),
        "num_test_files_used": len(dataset.file_records),
        "num_test_sequences": len(dataset),
        "num_test_seizure_sequences": int(np.sum(labels == 1)),
        "num_test_nonseizure_sequences": int(np.sum(labels == 0)),
        "excluded_files": dataset.excluded_files,
        "parameter_counts": count_parameters(model),
        "metrics": metrics,
    }
    metrics_path = output_dir / f"{args.run_name}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_payload, f, indent=2)

    qualitative_file = first_seizure_file(dataset)
    qualitative_dataset = CompactSpectrogramSequenceDataset(
        args.data_dir,
        [qualitative_file],
        common_channels,
        channel_labels_by_file,
        seq_len=args.seq_len,
        cache_files=True,
    )
    qualitative_loader = DataLoader(
        qualitative_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    _, q_probs, _, q_labels = evaluate_with_outputs(model, qualitative_loader, device)
    q_starts, _, q_patients, q_filenames = sequence_times_for_dataset(qualitative_dataset)
    figure_path = output_dir / f"{args.run_name}_probability_trace.png"
    plot_probability_trace(
        figure_path,
        q_starts,
        q_probs,
        q_labels,
        q_patients[0],
        q_filenames[0],
    )

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Test files input:", len(test_files))
    print("Test files used:", len(dataset.file_records))
    print("Excluded files:", len(dataset.excluded_files))
    print("Test sequences:", len(dataset))
    print("Test seizure sequences:", int(np.sum(labels == 1)))
    print("Test nonseizure sequences:", int(np.sum(labels == 0)))
    print("Metrics:", metrics)
    print("Saved metrics:", metrics_path)
    print("Saved predictions:", probs_path)
    print("Saved qualitative figure:", figure_path)


if __name__ == "__main__":
    main()
