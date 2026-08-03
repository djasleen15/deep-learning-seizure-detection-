"""Package processed train/validation .npz files into reusable tar shards.

The goal is to avoid repeatedly copying many loose files from Google Drive into
Colab. Archives are grouped by patient so Chunk 4/5 runs can extract only the
patient groups they need.

This script never packages the test split unless explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
    )
    parser.add_argument("--patients-per-shard", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate shards even if the tar file already exists.",
    )
    return parser.parse_args()


def patient_from_npz(path: Path) -> str:
    if not path.name.endswith("_windows.npz"):
        raise ValueError(f"Unexpected processed filename: {path.name}")

    return path.name.split("_", 1)[0]


def grouped_files(split_dir: Path) -> dict[str, list[Path]]:
    by_patient: dict[str, list[Path]] = {}

    for path in sorted(split_dir.glob("*.npz")):
        patient = patient_from_npz(path)
        by_patient.setdefault(patient, []).append(path)

    return by_patient


def chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def add_files_to_tar(tar: tarfile.TarFile, files: list[Path], split: str):
    for path in files:
        arcname = Path("processed_patient_splits") / split / path.name
        tar.add(path, arcname=arcname)


def main():
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "processed_dir": str(processed_dir),
        "patients_per_shard": args.patients_per_shard,
        "splits": args.splits,
        "shards": [],
        "test_split_packaged": "test" in args.splits,
    }

    for split in args.splits:
        split_dir = processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        by_patient = grouped_files(split_dir)
        patients = sorted(by_patient)

        for patient_group in chunks(patients, args.patients_per_shard):
            shard_name = f"{split}_{patient_group[0]}_to_{patient_group[-1]}.tar"
            shard_path = output_dir / shard_name

            files = []
            for patient in patient_group:
                files.extend(by_patient[patient])

            if shard_path.exists() and not args.overwrite:
                status = "skipped_existing"
                print(f"SKIP exists: {shard_path}")
            else:
                status = "saved"
                print(f"Creating {shard_path} with {len(files)} files")
                with tarfile.open(shard_path, mode="w") as tar:
                    add_files_to_tar(tar, files, split)

            manifest["shards"].append(
                {
                    "split": split,
                    "patients": patient_group,
                    "num_files": len(files),
                    "shard_path": str(shard_path),
                    "status": status,
                }
            )

    manifest_path = output_dir / "processed_shards_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print("Saved manifest:", manifest_path)
    print("Test split packaged:", manifest["test_split_packaged"])


if __name__ == "__main__":
    main()
