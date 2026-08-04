"""Quick profiler for lightweight CNN-LSTM training batches.

This is diagnostic-only. It does not change the training script, architecture,
or hyperparameters. It times three representative batches from the same compact
spectrogram dataset used by train_lightweight_cnn_lstm_full.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from seizure_detection.model import SmallCNNLSTM
from seizure_detection.resnet_input import (
    load_channel_labels_csv,
    load_common_channels,
    processed_npz_files,
)
from train_lightweight_cnn_lstm_full import (
    CompactSpectrogramSequenceDataset,
    filter_by_patients,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--common-channels-json", required=True)
    parser.add_argument("--channel-labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-patients", nargs="*")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--no-cache-files",
        action="store_true",
        help="Disable per-file array caching to match low-memory training runs.",
    )
    return parser.parse_args()


def make_manual_batch(dataset, start_idx, batch_size):
    xs = []
    ys = []
    end_idx = min(start_idx + batch_size, len(dataset))
    for idx in range(start_idx, end_idx):
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def timed_model_step(model, criterion, optimizer, batch_x, batch_y, device):
    timings = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    timings["device_transfer_ms"] = (time.perf_counter() - start) * 1000

    optimizer.zero_grad()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        with record_function("cnn_forward"):
            batch_size, seq_len, channels, height, width = batch_x.shape
            x = batch_x.reshape(batch_size * seq_len, channels, height, width)
            start = time.perf_counter()
            features = model.cnn(x)
            features = model.cnn_fc(features)
            timings["cnn_forward_ms"] = (time.perf_counter() - start) * 1000

        with record_function("lstm_classifier_forward"):
            start = time.perf_counter()
            features = features.reshape(batch_size, seq_len, -1)
            lstm_out, _ = model.lstm(features)
            final_out = lstm_out[:, -1, :]
            logits = model.classifier(final_out).squeeze(1)
            timings["lstm_classifier_forward_ms"] = (time.perf_counter() - start) * 1000

        with record_function("loss_backward"):
            loss = criterion(logits, batch_y)
            start = time.perf_counter()
            loss.backward()
            timings["backward_total_ms"] = (time.perf_counter() - start) * 1000

        with record_function("optimizer_step"):
            start = time.perf_counter()
            optimizer.step()
            timings["optimizer_step_ms"] = (time.perf_counter() - start) * 1000

    timings["total_step_ms"] = (time.perf_counter() - total_start) * 1000
    timings["loss"] = float(loss.detach().cpu())

    profiler_total_ms = sum(event.self_cpu_time_total for event in prof.key_averages()) / 1000
    timings["torch_profiler_self_cpu_ms"] = profiler_total_ms
    return timings


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_channels = load_common_channels(args.common_channels_json)
    channel_labels_by_file = load_channel_labels_csv(args.channel_labels_csv)
    train_files = processed_npz_files(args.processed_dir, "train")
    train_files = filter_by_patients(train_files, args.train_patients)

    dataset = CompactSpectrogramSequenceDataset(
        args.data_dir,
        train_files,
        common_channels,
        channel_labels_by_file,
        seq_len=args.seq_len,
        cache_files=not args.no_cache_files,
    )

    if len(dataset) < args.batch_size:
        raise ValueError("Dataset is smaller than one profiling batch.")

    batch_starts = {
        "early": 0,
        "middle": max(0, len(dataset) // 2 - args.batch_size // 2),
        "late": max(0, len(dataset) - args.batch_size),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNNLSTM(feature_dim=128, lstm_hidden=64).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    rows = []
    for label, start_idx in batch_starts.items():
        data_start = time.perf_counter()
        batch_x, batch_y = make_manual_batch(dataset, start_idx, args.batch_size)
        data_loading_ms = (time.perf_counter() - data_start) * 1000

        step_timings = timed_model_step(
            model,
            criterion,
            optimizer,
            batch_x,
            batch_y,
            device,
        )
        rows.append(
            {
                "batch_position": label,
                "start_sequence_idx": start_idx,
                "data_loading_ms": round(data_loading_ms, 2),
                "device_transfer_ms": round(step_timings["device_transfer_ms"], 2),
                "cnn_forward_ms": round(step_timings["cnn_forward_ms"], 2),
                "lstm_classifier_forward_ms": round(
                    step_timings["lstm_classifier_forward_ms"], 2
                ),
                "backward_total_ms": round(step_timings["backward_total_ms"], 2),
                "optimizer_step_ms": round(step_timings["optimizer_step_ms"], 2),
                "total_step_ms": round(step_timings["total_step_ms"], 2),
                "loss": round(step_timings["loss"], 6),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "lightweight_cnn_lstm_profile.csv"
    df.to_csv(csv_path, index=False)

    print("Dataset sequences:", len(dataset))
    print("Train files:", len(dataset.file_records))
    print("Device:", device)
    print("Cache files:", not args.no_cache_files)
    print(df.to_string(index=False))
    print("Saved profile CSV:", csv_path)

    means = df[
        [
            "data_loading_ms",
            "cnn_forward_ms",
            "lstm_classifier_forward_ms",
            "backward_total_ms",
            "optimizer_step_ms",
        ]
    ].mean()
    dominant = means.idxmax().replace("_ms", "")
    print("Dominant measured component:", dominant)


if __name__ == "__main__":
    main()
