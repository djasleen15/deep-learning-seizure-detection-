"""Train causal LSTM + classifier on precomputed frozen ResNet features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from seizure_detection.feature_input import (
    ResNetFeatureSequenceDataset,
    feature_npz_files,
    filter_feature_files_by_patients,
)
from seizure_detection.model import FeatureLSTM, count_parameters
from seizure_detection.training import evaluate_model, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="feature_lstm")
    parser.add_argument("--train-patients", nargs="*")
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--max-val-files", type=int)
    parser.add_argument(
        "--no-cache-files",
        action="store_true",
        help="Disable feature file caching to reduce RAM use.",
    )
    return parser.parse_args()


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

    train_files = feature_npz_files(args.feature_dir, "train")
    val_files = feature_npz_files(args.feature_dir, "val")
    train_files = filter_feature_files_by_patients(train_files, args.train_patients)

    if args.max_train_files is not None:
        train_files = train_files[: args.max_train_files]
    if args.max_val_files is not None:
        val_files = val_files[: args.max_val_files]

    train_dataset = ResNetFeatureSequenceDataset(
        train_files,
        seq_len=args.seq_len,
        cache_files=not args.no_cache_files,
    )
    val_dataset = ResNetFeatureSequenceDataset(
        val_files,
        seq_len=args.seq_len,
        cache_files=not args.no_cache_files,
    )

    train_labels = np.array(train_dataset.sequence_labels, dtype=int)
    val_labels = np.array(val_dataset.sequence_labels, dtype=int)

    print("Train feature files:", len(train_dataset.file_records))
    print("Validation feature files:", len(val_dataset.file_records))
    print("Train sequences:", len(train_dataset), class_counts(train_labels))
    print("Validation sequences:", len(val_dataset), class_counts(val_labels))
    print("Feature file caching:", not args.no_cache_files)
    print("Input shape per sequence: seq_len x 512")

    sampler = make_weighted_sampler(train_labels, args.seed)
    diagnostic = sampler_diagnostic(sampler, train_labels)
    print("WeightedRandomSampler diagnostic for one sampled epoch:", diagnostic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = FeatureLSTM(feature_dim=512, lstm_hidden=256).to(device)
    params = count_parameters(model)
    print("Device:", device)
    print("Parameter counts:", params)
    print("Frozen ResNet features: True")
    print("Optimizer: Adam")
    print("Learning rate:", args.learning_rate)
    print("Epochs:", args.epochs)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
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
