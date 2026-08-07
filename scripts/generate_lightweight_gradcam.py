"""Generate Grad-CAM figures for the finalized lightweight CNN-LSTM.

This script is read-only with respect to model/data decisions: it loads the
already-selected checkpoint and already-saved test predictions, selects
diagnostic examples, and writes interpretability figures.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from seizure_detection.model import SmallCNNLSTM
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
    parser.add_argument("--channel-labels-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="lightweight_cnn_lstm_gradcam")
    parser.add_argument("--seq-len", type=int, default=3)
    return parser.parse_args()


def parse_summary_seizures(summary_path: str | Path) -> list[dict[str, object]]:
    content = Path(summary_path).read_text()
    file_blocks = content.split("File Name:")[1:]
    rows = []

    for block in file_blocks:
        lines = block.strip().split("\n")
        filename = lines[0].strip()

        starts = re.findall(
            r"Seizure(?:\s\d+)?\sStart Time:\s*(\d+)\s*seconds",
            block,
        )
        ends = re.findall(
            r"Seizure(?:\s\d+)?\sEnd Time:\s*(\d+)\s*seconds",
            block,
        )

        for start, end in zip(starts, ends):
            rows.append(
                {
                    "filename": filename,
                    "start_sec": int(start),
                    "end_sec": int(end),
                }
            )

    return rows


def load_test_events(data_dir: Path, patients: list[str]) -> pd.DataFrame:
    event_rows = []
    for patient in patients:
        summary_path = data_dir / patient / f"{patient}-summary.txt"
        for row in parse_summary_seizures(summary_path):
            row["patient"] = patient
            event_rows.append(row)

    events = pd.DataFrame(event_rows)
    if events.empty:
        return events

    events = events[["patient", "filename", "start_sec", "end_sec"]]
    return events.sort_values(["patient", "filename", "start_sec"]).reset_index(drop=True)


def overlaps_interval(row, start_sec: float, end_sec: float) -> bool:
    return float(row["sequence_start_sec"]) < end_sec and float(row["sequence_end_sec"]) > start_sec


def compute_event_detection(preds: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_idx, event in events.iterrows():
        same_file = preds[
            (preds["patient"] == event["patient"])
            & (preds["filename"] == event["filename"])
        ]
        overlapping = same_file[
            same_file.apply(
                lambda row: overlaps_interval(row, event["start_sec"], event["end_sec"]),
                axis=1,
            )
        ]
        detected = bool((overlapping["predicted_label"].astype(int) == 1).any())
        rows.append(
            {
                "event_id": int(event_idx),
                "patient": event["patient"],
                "filename": event["filename"],
                "event_start_sec": int(event["start_sec"]),
                "event_end_sec": int(event["end_sec"]),
                "overlapping_sequences": int(len(overlapping)),
                "detected_by_at_least_one_sequence": detected,
                "max_overlapping_probability": (
                    float(overlapping["predicted_probability"].max())
                    if len(overlapping)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def select_examples(preds: pd.DataFrame) -> pd.DataFrame:
    fp_candidates = preds[
        (preds["true_label"].astype(int) == 0)
        & (preds["predicted_label"].astype(int) == 1)
    ].sort_values("predicted_probability", ascending=False)
    if fp_candidates.empty:
        raise RuntimeError("No false-positive examples found.")

    fp = fp_candidates.iloc[0].copy()
    fp_patient = fp["patient"]

    tp_candidates = preds[
        (preds["true_label"].astype(int) == 1)
        & (preds["predicted_label"].astype(int) == 1)
        & (preds["patient"] != fp_patient)
    ].sort_values("predicted_probability", ascending=False)
    if tp_candidates.empty:
        tp_candidates = preds[
            (preds["true_label"].astype(int) == 1)
            & (preds["predicted_label"].astype(int) == 1)
        ].sort_values("predicted_probability", ascending=False)
    if tp_candidates.empty:
        raise RuntimeError("No true-positive examples found.")
    tp = tp_candidates.iloc[0].copy()

    fn_candidates = preds[
        (preds["true_label"].astype(int) == 1)
        & (preds["predicted_label"].astype(int) == 0)
    ].sort_values("predicted_probability", ascending=True)
    if fn_candidates.empty:
        raise RuntimeError("No false-negative examples found.")
    fn = fn_candidates.iloc[0].copy()

    selected = pd.DataFrame([tp, fp, fn]).reset_index(drop=True)
    selected.insert(0, "example_type", ["true_positive", "false_positive", "false_negative"])
    return selected


def sequence_times_for_dataset(dataset: CompactSpectrogramSequenceDataset) -> pd.DataFrame:
    rows = []
    for sequence_row, (record_idx, start_idx) in enumerate(dataset.sequence_index):
        record = dataset.file_records[record_idx]
        data = np.load(record["path"], allow_pickle=True)
        window_times = data["window_times"]
        end_idx = start_idx + dataset.seq_len - 1
        rows.append(
            {
                "sequence_row": sequence_row,
                "patient": record["patient"],
                "filename": record["filename"],
                "sequence_start_sec": float(window_times[start_idx][0]),
                "sequence_end_sec": float(window_times[end_idx][1]),
            }
        )
    return pd.DataFrame(rows)


class GradCAM:
    def __init__(self, model: SmallCNNLSTM, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, sequence: torch.Tensor) -> tuple[float, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(sequence)
        logits.squeeze().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cams = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cams = F.interpolate(
            cams,
            size=sequence.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        cams_np = cams.cpu().numpy()
        for idx in range(cams_np.shape[0]):
            cam = cams_np[idx]
            cam_min = cam.min()
            cam_max = cam.max()
            cams_np[idx] = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        prob = torch.sigmoid(logits.detach()).item()
        return prob, cams_np


def get_sequence_by_row(dataset: CompactSpectrogramSequenceDataset, sequence_row: int):
    x, y = dataset[sequence_row]
    return x.unsqueeze(0), int(y.item())


def plot_gradcam_example(
    output_path: Path,
    sequence: torch.Tensor,
    cams: np.ndarray,
    example: pd.Series,
    common_channel_count: int,
):
    seq = sequence.squeeze(0).detach().cpu().numpy()
    fig, axes = plt.subplots(1, seq.shape[0], figsize=(13, 4), sharey=True)
    if seq.shape[0] == 1:
        axes = [axes]

    for window_idx, ax in enumerate(axes):
        image = seq[window_idx, 0]
        cam = cams[window_idx]
        ax.imshow(image, aspect="auto", origin="lower", cmap="viridis")
        ax.imshow(cam, aspect="auto", origin="lower", cmap="magma", alpha=0.45)
        ax.set_title(f"Window {window_idx + 1}")
        ax.set_xlabel("STFT time bin")

    axes[0].set_ylabel(f"Channel-frequency row\n({common_channel_count} channels x freq bins)")
    fig.suptitle(
        (
            f"{example['example_type']} | {example['patient']} {example['filename']} | "
            f"{example['sequence_start_sec']:.1f}-{example['sequence_end_sec']:.1f}s | "
            f"p={example['predicted_probability']:.3f}"
        ),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_channels = load_common_channels(args.common_channels_json)
    channel_labels_by_file = load_channel_labels_csv(args.channel_labels_csv)
    test_files = processed_npz_files(args.processed_dir, "test")
    preds = pd.read_csv(args.predictions_csv)
    preds = preds.reset_index().rename(columns={"index": "sequence_row"})

    dataset = CompactSpectrogramSequenceDataset(
        data_dir,
        test_files,
        common_channels,
        channel_labels_by_file,
        seq_len=args.seq_len,
        cache_files=True,
    )
    dataset_times = sequence_times_for_dataset(dataset)
    if len(dataset_times) != len(preds):
        raise RuntimeError(
            f"Dataset has {len(dataset_times)} sequences but predictions CSV has {len(preds)}."
        )

    selected = select_examples(preds)
    selected_path = output_dir / f"{args.run_name}_selected_examples.csv"
    selected.to_csv(selected_path, index=False)

    events = load_test_events(data_dir, ["chb22", "chb23", "chb24"])
    event_detection = compute_event_detection(preds, events)
    event_detection_path = output_dir / f"{args.run_name}_event_detection.csv"
    event_detection.to_csv(event_detection_path, index=False)
    detected_events = int(event_detection["detected_by_at_least_one_sequence"].sum())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNNLSTM(feature_dim=128, lstm_hidden=64).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = model.cnn[4]
    gradcam = GradCAM(model, target_layer)
    figure_paths = []
    try:
        for _, example in selected.iterrows():
            sequence, _label = get_sequence_by_row(dataset, int(example["sequence_row"]))
            sequence = sequence.to(device)
            prob, cams = gradcam(sequence)

            output_path = output_dir / f"{args.run_name}_{example['example_type']}.png"
            plot_gradcam_example(
                output_path,
                sequence.cpu(),
                cams,
                example,
                common_channel_count=len(common_channels),
            )
            figure_paths.append(str(output_path))
            print(
                f"{example['example_type']}: "
                f"{example['patient']} {example['filename']} "
                f"{example['sequence_start_sec']:.1f}-{example['sequence_end_sec']:.1f}s "
                f"prob={prob:.4f} saved={output_path}"
            )
    finally:
        gradcam.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "target_layer": "SmallCNNLSTM.cnn[4]",
        "common_channel_count": len(common_channels),
        "num_test_events": int(len(events)),
        "num_detected_test_events": detected_events,
        "event_level_detection_rate": (
            detected_events / len(events) if len(events) else None
        ),
        "selected_examples_csv": str(selected_path),
        "event_detection_csv": str(event_detection_path),
        "figure_paths": figure_paths,
    }
    summary_path = output_dir / f"{args.run_name}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nGrad-CAM complete.")
    print("Device:", device)
    print("Target layer: SmallCNNLSTM.cnn[4]")
    print("Selected examples:", selected_path)
    print("Event detection CSV:", event_detection_path)
    print(
        "Event-level detection:",
        f"{detected_events}/{len(events)}",
        f"({summary['event_level_detection_rate']:.4f})" if len(events) else "",
    )
    print("Summary JSON:", summary_path)


if __name__ == "__main__":
    main()
