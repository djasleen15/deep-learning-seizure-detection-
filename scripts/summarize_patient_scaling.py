"""Summarize patient-count scaling runs into one CSV and figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        metavar=(
            "NUM_PATIENTS",
            "NUM_TRAIN_SEQUENCES",
            "NUM_SEIZURE_SEQUENCES",
            "WALL_CLOCK_SECONDS",
            "METRICS_CSV",
        ),
        help=(
            "Add one scaling run. Repeat for each patient count. "
            "Example: --run 6 51648 625 3600 /path/metrics.csv"
        ),
    )
    return parser.parse_args()


def summarize_run(raw_run):
    num_patients, num_sequences, num_seizure_sequences, wall_clock_seconds, metrics_csv = raw_run
    metrics = pd.read_csv(metrics_csv)
    final = metrics.iloc[-1]
    best_idx = metrics["f1"].idxmax()
    best = metrics.loc[best_idx]

    return {
        "num_train_patients": int(num_patients),
        "num_train_sequences": int(num_sequences),
        "num_seizure_sequences": int(num_seizure_sequences),
        "final_epoch": int(final["epoch"]),
        "final_epoch_F1": float(final["f1"]),
        "final_epoch_sensitivity": float(final["sensitivity_recall"]),
        "final_epoch_specificity": float(final["specificity"]),
        "final_epoch_precision": float(final["precision"]),
        "best_epoch": int(best["epoch"]),
        "best_epoch_F1": float(best["f1"]),
        "wall_clock_seconds": float(wall_clock_seconds),
        "metrics_csv": metrics_csv,
    }


def write_summary_csv(path: Path, rows: list[dict]):
    fields = [
        "num_train_patients",
        "num_train_sequences",
        "num_seizure_sequences",
        "final_epoch",
        "final_epoch_F1",
        "final_epoch_sensitivity",
        "final_epoch_specificity",
        "final_epoch_precision",
        "best_epoch",
        "best_epoch_F1",
        "wall_clock_seconds",
        "metrics_csv",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_scaling(path: Path, rows: list[dict]):
    rows = sorted(rows, key=lambda row: row["num_train_patients"])
    x = [row["num_train_patients"] for row in rows]
    final_f1 = [row["final_epoch_F1"] for row in rows]
    best_f1 = [row["best_epoch_F1"] for row in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(x, final_f1, marker="o", label="Final epoch F1")
    plt.plot(x, best_f1, marker="s", linestyle="--", label="Best epoch F1")
    plt.xlabel("Number of training patients")
    plt.ylabel("Validation F1")
    plt.title("Patient-count scaling on held-out validation patients")
    plt.xticks(x)
    plt.ylim(0, max(best_f1 + final_f1 + [0.1]) * 1.15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    if not args.run:
        raise ValueError("At least one --run entry is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_run(run) for run in args.run]
    rows = sorted(rows, key=lambda row: row["num_train_patients"])

    summary_path = output_dir / "patient_count_scaling_summary.csv"
    figure_path = output_dir / "patient_count_scaling_f1.png"

    write_summary_csv(summary_path, rows)
    plot_scaling(figure_path, rows)

    print("Saved summary:", summary_path)
    print("Saved figure:", figure_path)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
