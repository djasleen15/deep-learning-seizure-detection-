"""Process all seizure-containing chb01 EDF files into STFT windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from seizure_detection.data_processing import (
    load_all_annotations,
    process_one_edf,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to chb-mit-data folder")
    parser.add_argument("--output-dir", default="processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_all_annotations(data_dir, ["chb01"])
    seizure_files = [
        filename
        for filename, info in annotations["chb01"].items()
        if info["num_seizures"] > 0
    ]

    all_specs = []
    all_labels = []
    all_file_ids = []
    summaries = []
    kept_freqs = None
    kept_times = None

    for filename in seizure_files:
        print(f"Processing {filename}...")
        specs, labels, _, kept_freqs, kept_times, summary = process_one_edf(
            data_dir,
            annotations,
            "chb01",
            filename,
        )
        all_specs.append(specs)
        all_labels.append(labels)
        all_file_ids.extend([filename] * len(labels))
        summaries.append(summary)
        print(summary)

    x = np.concatenate(all_specs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    file_ids = np.array(all_file_ids)
    summary_df = pd.DataFrame(summaries)

    np.savez_compressed(
        output_dir / "chb01_all_seizure_files_windows.npz",
        X=x,
        y=y,
        file_ids=file_ids,
        kept_freqs=kept_freqs,
        kept_times=kept_times,
    )
    summary_df.to_csv(output_dir / "chb01_all_seizure_files_summary.csv", index=False)

    print("Saved processed windows and summary.")
    print("X shape:", x.shape)
    print("Seizure windows:", int(np.sum(y == 1)))
    print("Non-seizure windows:", int(np.sum(y == 0)))


if __name__ == "__main__":
    main()

