"""Plotting helpers for progress-report figures."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def save_class_balance_plot(y, save_path, title):
    """Save a seizure/non-seizure class balance bar chart."""
    counts = [int(np.sum(y == 0)), int(np.sum(y == 1))]
    labels_text = ["Non-seizure", "Seizure"]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(labels_text, counts, color=["#4C78A8", "#F58518"])

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.ylabel("Number of 4-second windows")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_validation_probability_plot(
    val_probs,
    val_labels,
    save_path,
    seq_len: int = 3,
    window_sec: int = 4,
    title: str = "CNN-LSTM validation predictions",
):
    """Save model predicted seizure probability over recording time."""
    sequence_start_times = np.arange(len(val_probs)) * window_sec
    sequence_end_times = sequence_start_times + seq_len * window_sec
    sequence_mid_times = (sequence_start_times + sequence_end_times) / 2

    plt.figure(figsize=(14, 4))
    plt.plot(sequence_mid_times, val_probs, label="Predicted seizure probability")
    plt.scatter(
        sequence_mid_times[val_labels == 1],
        val_probs[val_labels == 1],
        color="red",
        label="True seizure sequences",
        zorder=3,
    )
    plt.axhline(0.5, color="black", linestyle="--", label="Decision threshold")
    plt.xlabel("Time in validation recording (s)")
    plt.ylabel("Predicted seizure probability")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

