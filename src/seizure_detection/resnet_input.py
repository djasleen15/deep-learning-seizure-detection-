"""On-the-fly ResNet input dataset for compact processed CHB-MIT files."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
import pyedflib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_common_channels(path: str | Path) -> list[str]:
    """Load the saved train+validation common channel list."""
    with Path(path).open("r") as f:
        payload = json.load(f)
    return list(payload["common_channels"])


def processed_npz_files(processed_dir: str | Path, split: str) -> list[Path]:
    """Return compact per-EDF processed files for one split."""
    return sorted((Path(processed_dir) / split).glob("*.npz"))


def load_channel_labels_csv(path: str | Path) -> dict[tuple[str, str], list[str]]:
    """Load saved EDF channel labels from inspect_common_channels.py output."""
    labels_by_file = {}

    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_by_file[(row["patient"], row["filename"])] = row[
                "channel_labels"
            ].split("|")

    return labels_by_file


def channel_indices_for_file(
    data_dir: str | Path,
    patient: str,
    filename: str,
    common_channels: list[str],
    channel_labels_by_file: dict[tuple[str, str], list[str]] | None = None,
) -> tuple[list[int] | None, list[str], list[str]]:
    """Map common channel labels onto a raw EDF file's channel order."""
    key = (patient, filename)
    if channel_labels_by_file is None or key not in channel_labels_by_file:
        labels = read_edf_channel_labels(Path(data_dir) / patient / filename)
    else:
        labels = list(channel_labels_by_file[key])

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    missing = [label for label in common_channels if label not in label_to_index]

    if missing:
        return None, missing, labels

    return [label_to_index[label] for label in common_channels], [], labels


def read_edf_channel_labels(path: str | Path) -> list[str]:
    """Read EDF signal labels without loading signal samples."""
    reader = pyedflib.EdfReader(str(path))
    try:
        return list(reader.getSignalLabels())
    finally:
        reader.close()


def resize_window_to_resnet_rgb(window: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Convert one standardized spectrogram window to 3x224x224.

    Input shape:
      common_channels x freq_bins x time_bins

    The common channel and frequency dimensions are flattened into one image
    height dimension. Bilinear interpolation resizes the resulting single-channel
    image to 224x224, then the image is replicated across 3 channels for
    ResNet-18's expected RGB input format.
    """
    n_channels, n_freqs, n_times = window.shape
    image = torch.tensor(
        window.reshape(1, 1, n_channels * n_freqs, n_times),
        dtype=torch.float32,
    )
    image = F.interpolate(
        image,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return image.squeeze(0).repeat(3, 1, 1)


class ResNetSequenceDataset(Dataset):
    """Lazy sequence dataset that resizes compact spectrograms on the fly.

    The dataset indexes sequences within each EDF separately. Therefore sequence
    construction never crosses file boundaries.
    """

    def __init__(
        self,
        data_dir: str | Path,
        npz_files: list[str | Path],
        common_channels: list[str],
        seq_len: int = 3,
        image_size: int = 224,
        max_files: int | None = None,
        channel_labels_by_file: dict[tuple[str, str], list[str]] | None = None,
        cache_files: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.common_channels = list(common_channels)
        self.seq_len = seq_len
        self.image_size = image_size
        self.channel_labels_by_file = channel_labels_by_file
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
                self.excluded_files.append(
                    {
                        "path": str(npz_path),
                        "patient": patient,
                        "filename": filename,
                        "missing_channels": [],
                        "raw_channel_count": len(raw_labels),
                        "reason": "too_few_windows",
                    }
                )
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
                self.sequence_index.append((record_idx, start_idx))
                end_idx = start_idx + seq_len
                label = 1 if np.any(y[start_idx:end_idx] == 1) else 0
                self.sequence_labels.append(label)

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, idx: int):
        record_idx, start_idx = self.sequence_index[idx]
        record = self.file_records[record_idx]

        if self.cache_files:
            if record_idx not in self._file_cache:
                data = np.load(record["path"], allow_pickle=True)
                self._file_cache[record_idx] = {
                    "X": data["X"],
                    "y": data["y"],
                }

            data = self._file_cache[record_idx]
        else:
            data = np.load(record["path"], allow_pickle=True)

        x = data["X"]
        y = data["y"]

        end_idx = start_idx + self.seq_len
        windows = x[start_idx:end_idx, record["channel_indices"], :, :]
        label = 1 if np.any(y[start_idx:end_idx] == 1) else 0

        sequence = torch.stack(
            [
                resize_window_to_resnet_rgb(window, image_size=self.image_size)
                for window in windows
            ],
            dim=0,
        )

        return sequence, torch.tensor(label, dtype=torch.float32)
