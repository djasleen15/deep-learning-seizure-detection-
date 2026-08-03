"""Datasets for precomputed ResNet feature files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def feature_npz_files(feature_dir: str | Path, split: str) -> list[Path]:
    """Return per-EDF feature files for one split."""
    return sorted((Path(feature_dir) / split).glob("*_features.npz"))


def patient_from_feature_npz(path: Path) -> str:
    data = np.load(path, allow_pickle=True)
    return str(data["patient_id"])


def filter_feature_files_by_patients(
    files: list[Path],
    patients: list[str] | None,
) -> list[Path]:
    if not patients:
        return files

    patients = set(patients)
    return [path for path in files if patient_from_feature_npz(path) in patients]


class ResNetFeatureSequenceDataset(Dataset):
    """Build 3-window sequences from per-window 512-dim ResNet features."""

    def __init__(
        self,
        feature_files: list[str | Path],
        seq_len: int = 3,
        max_files: int | None = None,
        cache_files: bool = True,
    ):
        self.seq_len = seq_len
        self.cache_files = cache_files
        self.file_records = []
        self.sequence_index = []
        self.sequence_labels = []
        self._file_cache = {}

        selected_files = [Path(path) for path in feature_files]
        if max_files is not None:
            selected_files = selected_files[:max_files]

        for feature_path in selected_files:
            data = np.load(feature_path, allow_pickle=True)
            y = data["y"]

            if len(y) < seq_len:
                continue

            record_idx = len(self.file_records)
            self.file_records.append(
                {
                    "path": feature_path,
                    "patient": str(data["patient_id"]),
                    "filename": str(data["file_id"]),
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
                    "features": data["features"],
                    "y": data["y"],
                }
            data = self._file_cache[record_idx]
        else:
            data = np.load(record["path"], allow_pickle=True)

        end_idx = start_idx + self.seq_len
        features = data["features"][start_idx:end_idx]
        y = data["y"]
        label = 1 if np.any(y[start_idx:end_idx] == 1) else 0

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )
