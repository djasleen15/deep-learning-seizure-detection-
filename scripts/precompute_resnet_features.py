"""Precompute frozen ResNet-18 features for processed EEG spectrogram windows.

This is the deadline-safe Chunk 4 path: ResNet-18 is used as a frozen ImageNet
feature extractor, and each 4-second window is saved as one 512-dim feature
vector. The script works file-by-file with skip-if-exists behavior, so it is
safe to rerun after a runtime disconnect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from seizure_detection.resnet_input import (
    channel_indices_for_file,
    load_channel_labels_csv,
    load_common_channels,
    processed_npz_files,
    resize_window_to_resnet_rgb,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--common-channels-json", required=True)
    parser.add_argument("--channel-labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet weights. Use only if network access is unavailable.",
    )
    return parser.parse_args()


class WindowResNetInputDataset(Dataset):
    """One processed EDF file as per-window ResNet input tensors."""

    def __init__(self, windows, image_size):
        self.windows = windows
        self.image_size = image_size

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return resize_window_to_resnet_rgb(self.windows[idx], image_size=self.image_size)


def output_path_for(input_path: Path, output_dir: Path, split: str) -> Path:
    name = input_path.name.replace("_windows.npz", "_features.npz")
    return output_dir / split / name


def save_features(
    output_path: Path,
    features: np.ndarray,
    source_data,
    common_channels: list[str],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features.astype(np.float32),
        y=source_data["y"],
        window_times=source_data["window_times"],
        patient_id=source_data["patient_id"],
        file_id=source_data["file_id"],
        fs=source_data["fs"],
        duration_sec=source_data["duration_sec"],
        common_channels=np.array(common_channels),
        feature_dim=np.array(features.shape[1]),
    )


def extract_one_file(
    npz_path: Path,
    output_path: Path,
    args,
    model,
    device,
    common_channels,
    channel_labels_by_file,
):
    data = np.load(npz_path, allow_pickle=True)
    patient = str(data["patient_id"])
    filename = str(data["file_id"])

    channel_indices, missing, raw_labels = channel_indices_for_file(
        args.data_dir,
        patient,
        filename,
        common_channels,
        channel_labels_by_file=channel_labels_by_file,
    )

    if missing:
        return {
            "status": "excluded_missing_channels",
            "path": str(npz_path),
            "patient": patient,
            "filename": filename,
            "missing_channels": missing,
            "raw_channel_count": len(raw_labels),
        }

    windows = data["X"][:, channel_indices, :, :]
    dataset = WindowResNetInputDataset(windows, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    all_features = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            batch_features = model(batch_x)
            all_features.append(batch_features.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    save_features(output_path, features, data, common_channels)

    return {
        "status": "saved",
        "path": str(npz_path),
        "output_path": str(output_path),
        "patient": patient,
        "filename": filename,
        "num_windows": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_seizure_windows": int(np.sum(data["y"] == 1)),
        "num_nonseizure_windows": int(np.sum(data["y"] == 0)),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_channels = load_common_channels(args.common_channels_json)
    channel_labels_by_file = load_channel_labels_csv(args.channel_labels_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = None if args.no_pretrained else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    print("Device:", device)
    print("Frozen ResNet-18 feature extractor: True")
    print("Common channel count:", len(common_channels))
    print("Output dir:", output_dir)

    rows = []
    for split in args.splits:
        files = processed_npz_files(args.processed_dir, split)
        if args.max_files is not None:
            files = files[: args.max_files]

        print(f"\nSplit {split}: {len(files)} processed files")

        for idx, npz_path in enumerate(files, start=1):
            out_path = output_path_for(npz_path, output_dir, split)
            if out_path.exists() and not args.overwrite:
                print(f"[{idx}/{len(files)}] SKIP exists: {out_path.name}")
                rows.append(
                    {
                        "status": "skipped_existing",
                        "path": str(npz_path),
                        "output_path": str(out_path),
                    }
                )
                continue

            print(f"[{idx}/{len(files)}] Processing {split}/{npz_path.name}")
            row = extract_one_file(
                npz_path,
                out_path,
                args,
                model,
                device,
                common_channels,
                channel_labels_by_file,
            )
            rows.append(row)
            print(row)

    manifest_path = output_dir / "resnet_feature_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(rows, f, indent=2)

    saved = [row for row in rows if row["status"] in {"saved", "skipped_existing"}]
    excluded = [row for row in rows if row["status"].startswith("excluded")]
    total_windows = sum(int(row.get("num_windows", 0)) for row in saved)

    print("\nFeature precompute summary")
    print("Files saved or seen:", len(saved))
    print("Files excluded:", len(excluded))
    print("Total windows represented in newly saved rows:", total_windows)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
