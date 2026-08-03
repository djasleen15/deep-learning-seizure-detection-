"""Train the lightweight CNN-LSTM on processed patient-level split files.

This is the CPU-friendly primary-model path for deadline runs. It reuses the
progress-report architecture but reads compact per-EDF spectrogram .npz files,
standardizes channels using saved channel labels, and builds sequences lazily so
the full split does not need to be concatenated into one large array.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from seizure_detection.model import SmallCNNLSTM, count_parameters
from seizure_detection.resnet_input import (
    channel_indices_for_file,
    load_channel_labels_csv,
    load_common_channels,
    processed_npz_files,
)
from seizure_detection.training import evaluate_model, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--common-channels-json", required=True)
    parser.add_argument("--channel-labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="lightweight_cnn_lstm")
    parser.add_argument("--train-patients", nargs="*")
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--max-val-files", type=int)
    parser.add_argument(
        "--no-cache-files",
        action="store_true",
        help="Disable per-file array caching to reduce RAM use.",
    )
    return parser.parse_args()


def patient_from_processed_name(path: Path) -> str:
    if not path.name.endswith("_windows.npz"):
        raise ValueError(f"Unexpected processed filename: {path.name}")
    return path.name.split("_", 1)[0]


def filter_by_patients(files: list[Path], patients: list[str] | None) -> list[Path]:
    if not patients:
        return files
    patients = set(patients)
    return [path for path in files if patient_from_processed_name(path) in patients]


class CompactSpectrogramSequenceDataset(Dataset):
    """Lazy sequence dataset for compact STFT spectrograms.

    Each returned sample has shape:
      seq_len x 1 x (common_channels * freq_bins) x time_bins
    """

    def __init__(
        self,
        data_dir: str | Path,
        npz_files: list[str | Path],
        common_channels: list[str],
        channel_labels_by_file: dict[tuple[str, str], list[str]],
        seq_len: int = 3,
        max_files: int | None = None,
        cache_files: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.common_channels = common_channels
        self.channel_labels_by_file = channel_labels_by_file
        self.seq_len = seq_len
        self.cache_files = cache_files
        self.file_records = []
        self.sequence_index = []
        self.sequence_labels = []
        self.excluded_files = []
        self._file_cache = {}

        selected_files = [Path(path) for path in npz_files]
        if max_files is not None:
            selected_files = selected_files[:max_files]

        for npz_path in selected_files:
            data = np.load(npz_path, allow_pickle=True)
            patient = str(data["patient_id"])
            filename = str(data["file_id"])

            channel_indices, missing, raw_labels = channel_indices_for_file(
                self.data_dir,
                patient,
                filename,
                self.common_channels,
                channel_labels_by_file=self.channel_labels_by_file,
            )

            if missing:
                self.excluded_files.append(
                    {
                        "path": str(npz_path),
                        "patient": patient,
                        "filename": filename,
                        "missing_channels": missing,
                        "raw_channel_count": len(raw_labels),
                    }
                )
                continue

            y = data["y"]
            if len(y) < seq_len:
                continue

            record_idx = len(self.file_records)
            self.file_records.append(
                {
                    "path": npz_path,
                    "patient": patient,
                    "filename": filename,
                    "channel_indices": channel_indices,
                    "num_windows": len(y),
                }
            )

            for start_idx in range(0, len(y) - seq_len + 1):
                end_idx = start_idx + seq_len
                self.sequence_index.append((record_idx, start_idx))
                self.sequence_labels.append(1 if np.any(y[start_idx:end_idx] == 1) else 0)

    def __len__(self):
        return len(self.sequence_index)

    def _load_record_arrays(self, record_idx: int):
        if self.cache_files:
            if record_idx not in self._file_cache:
                record = self.file_records[record_idx]
                data = np.load(record["path"], allow_pickle=True)
                self._file_cache[record_idx] = {"X": data["X"], "y": data["y"]}
            return self._file_cache[record_idx]

        record = self.file_records[record_idx]
        data = np.load(record["path"], allow_pickle=True)
        return {"X": data["X"], "y": data["y"]}

    def __getitem__(self, idx):
        record_idx, start_idx = self.sequence_index[idx]
        record = self.file_records[record_idx]
        data = self._load_record_arrays(record_idx)

        end_idx = start_idx + self.seq_len
        windows = data["X"][start_idx:end_idx, record["channel_indices"], :, :]
        y = data["y"]
        label = 1 if np.any(y[start_idx:end_idx] == 1) else 0

        seq_len, num_channels, num_freqs, num_times = windows.shape
        windows = windows.reshape(seq_len, 1, num_channels * num_freqs, num_times)

        return (
            torch.tensor(windows, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "nonseizure_sequences": int(np.sum(labels == 0)),
        "seizure_sequences": int(np.sum(labels == 1)),
    }


def make_weighted_sampler(labels: np.ndarray, seed: int):
    counts = np.bincount(labels.astype(int), minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"Weighted sampler needs both classes; got counts={counts.tolist()}")

    class_weights = 1.0 / counts
    sample_weights = class_weights[labels.astype(int)]
    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def sampler_diagnostic(sampler: WeightedRandomSampler, labels: np.ndarray) -> dict[str, float]:
    sampled_indices = list(iter(sampler))
    sampled_counts = np.bincount(sampled_indices, minlength=len(labels))
    seizure_counts = sampled_counts[labels == 1]
    nonseizure_counts = sampled_counts[labels == 0]

    return {
        "seizure_min": int(seizure_counts.min()),
        "seizure_max": int(seizure_counts.max()),
        "seizure_mean": round(float(seizure_counts.mean()), 2),
        "nonseizure_min": int(nonseizure_counts.min()),
        "nonseizure_max": int(nonseizure_counts.max()),
        "nonseizure_mean": round(float(nonseizure_counts.mean()), 2),
    }


def save_checkpoint(output_dir, run_name, epoch, model, optimizer, history, args, params):
    checkpoint_path = output_dir / f"{run_name}_epoch_{epoch:02d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
            "parameter_counts": params,
        },
        checkpoint_path,
    )
    return checkpoint_path


def write_history_csv(output_dir: Path, run_name: str, history: list[dict]):
    path = output_dir / f"{run_name}_metrics.csv"
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "TP",
        "TN",
        "FP",
        "FN",
        "sensitivity_recall",
        "specificity",
        "precision",
        "f1",
        "accuracy",
    ]

    with path.open("w") as f:
        f.write(",".join(fields) + "\n")
        for row in history:
            f.write(",".join(str(row.get(field, "")) for field in fields) + "\n")

    return path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_channels = load_common_channels(args.common_channels_json)
    channel_labels_by_file = load_channel_labels_csv(args.channel_labels_csv)

    train_files = processed_npz_files(args.processed_dir, "train")
    val_files = processed_npz_files(args.processed_dir, "val")
    train_files = filter_by_patients(train_files, args.train_patients)

    if args.max_train_files is not None:
        train_files = train_files[: args.max_train_files]
    if args.max_val_files is not None:
        val_files = val_files[: args.max_val_files]

    train_dataset = CompactSpectrogramSequenceDataset(
        args.data_dir,
        train_files,
        common_channels,
        channel_labels_by_file,
        seq_len=args.seq_len,
        cache_files=not args.no_cache_files,
    )
    val_dataset = CompactSpectrogramSequenceDataset(
        args.data_dir,
        val_files,
        common_channels,
        channel_labels_by_file,
        seq_len=args.seq_len,
        cache_files=not args.no_cache_files,
    )

    if train_dataset.excluded_files or val_dataset.excluded_files:
        excluded_path = output_dir / f"{args.run_name}_excluded_files.json"
        with excluded_path.open("w") as f:
            json.dump(
                {
                    "train": train_dataset.excluded_files,
                    "val": val_dataset.excluded_files,
                },
                f,
                indent=2,
            )
        print("Excluded files saved:", excluded_path)
    else:
        print("Excluded files: 0")

    train_labels = np.array(train_dataset.sequence_labels, dtype=int)
    val_labels = np.array(val_dataset.sequence_labels, dtype=int)

    print("Common channel count:", len(common_channels))
    print("Train files:", len(train_dataset.file_records))
    print("Validation files:", len(val_dataset.file_records))
    print("Train sequences:", len(train_dataset), class_counts(train_labels))
    print("Validation sequences:", len(val_dataset), class_counts(val_labels))
    print("Sequences cross file boundaries: False")
    print("Dataset file caching:", not args.no_cache_files)
    print("Input shape per sequence: seq_len x 1 x channel_freq x time")

    sampler = make_weighted_sampler(train_labels, args.seed)
    diagnostic = sampler_diagnostic(sampler, train_labels)
    print("WeightedRandomSampler diagnostic for one sampled epoch:", diagnostic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SmallCNNLSTM(feature_dim=128, lstm_hidden=64).to(device)
    params = count_parameters(model)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Device:", device)
    print("Parameter counts:", params)
    print("Optimizer: Adam")
    print("Learning rate:", args.learning_rate)
    print("Epochs:", args.epochs)

    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _, _, _ = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
        }
        row.update(val_metrics)
        history.append(row)

        checkpoint_path = save_checkpoint(
            output_dir,
            args.run_name,
            epoch,
            model,
            optimizer,
            history,
            args,
            params,
        )
        metrics_path = write_history_csv(output_dir, args.run_name, history)

        print(
            f"Epoch {epoch}: "
            f"train_loss={row['train_loss']:.6f}, "
            f"val_loss={row['val_loss']:.6f}, "
            f"val_f1={row['f1']:.4f}, "
            f"val_sens={row['sensitivity_recall']:.4f}, "
            f"val_spec={row['specificity']:.4f}, "
            f"checkpoint={checkpoint_path.name}"
        )

    elapsed_minutes = (time.time() - start_time) / 60
    best_epoch = max(history, key=lambda row: row["f1"])
    final_epoch = history[-1]

    print("Metrics CSV:", metrics_path)
    print("Final epoch metrics:", final_epoch)
    print("Best validation F1 epoch:", best_epoch)
    print("Total training time minutes:", round(elapsed_minutes, 2))


if __name__ == "__main__":
    main()
