"""Process CHB-MIT EDF files into patient-level split datasets.

Important:
- Train patients: chb01-chb18
- Validation patients: chb19-chb21
- Test patients: chb22-chb24

This script saves one processed .npz per EDF file instead of concatenating all
files into one array. CHB-MIT files can have different channel counts, so
concatenation is deferred until a later model-input standardization step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from seizure_detection.data_processing import load_all_annotations, process_one_edf
from seizure_detection.splits import PATIENT_SPLITS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to chb-mit-data folder")
    parser.add_argument("--output-dir", required=True, help="Where processed .npz files are saved")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
        help="Splits to process. Default avoids test until final evaluation prep.",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        help="Optional explicit patients for a single split, e.g. chb01 chb02 chb03",
    )
    parser.add_argument(
        "--include-all-files",
        action="store_true",
        help="Process all EDF files, not only files with seizures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list selected files; do not load/process EDFs or save arrays.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even if their per-EDF .npz already exists.",
    )
    return parser.parse_args()


def patient_edf_files(data_dir, patient):
    patient_dir = data_dir / patient
    return sorted(path.name for path in patient_dir.glob("*.edf"))


def selected_files_for_patient(data_dir, annotations, patient, include_all_files):
    available_edfs = patient_edf_files(data_dir, patient)

    if include_all_files:
        selected = available_edfs
    else:
        selected = [
            filename
            for filename in available_edfs
            if annotations[patient].get(filename, {}).get("num_seizures", 0) > 0
        ]

    return available_edfs, selected


def safe_file_stem(patient, filename):
    return f"{patient}_{filename.replace('.edf', '')}"


def save_processed_edf(
    output_dir,
    split_name,
    patient,
    filename,
    specs,
    labels,
    window_times,
    kept_freqs,
    kept_times,
    summary,
):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    save_path = split_dir / f"{safe_file_stem(patient, filename)}_windows.npz"

    np.savez_compressed(
        save_path,
        X=specs,
        y=labels,
        window_times=np.array(window_times),
        kept_freqs=kept_freqs,
        kept_times=kept_times,
        patient_id=np.array(patient),
        file_id=np.array(filename),
        fs=np.array(summary["fs"]),
        num_channels=np.array(summary["num_channels"]),
        duration_sec=np.array(summary["duration_sec"]),
    )

    return save_path


def resolve_patients_for_split(args, split_name):
    if args.patients:
        if len(args.splits) != 1:
            raise ValueError("--patients can only be used with one split at a time")

        invalid_patients = [
            patient
            for patient in args.patients
            if patient not in PATIENT_SPLITS[split_name]
        ]

        if invalid_patients:
            raise ValueError(
                f"Patients {invalid_patients} do not belong to split {split_name}"
            )

        return args.patients

    return PATIENT_SPLITS[split_name]


def process_split(data_dir, output_dir, split_name, patients, include_all_files, dry_run, overwrite):
    print(f"\n==============================")
    print(f"Processing split: {split_name}")
    print(f"Patients: {patients}")
    print(f"==============================")

    annotations = load_all_annotations(data_dir, patients)

    failed_files = []
    processed_summaries = []

    total_windows = 0
    total_seizure = 0
    total_nonseizure = 0

    for patient in patients:
        available_edfs, filenames = selected_files_for_patient(
            data_dir,
            annotations,
            patient,
            include_all_files,
        )

        if not available_edfs:
            print(f"WARNING: no EDF files found for {patient}")
            continue

        print(f"\n{patient}: {len(filenames)} files selected out of {len(available_edfs)} EDF files")

        if dry_run:
            for filename in filenames:
                num_seizures = annotations[patient].get(filename, {}).get("num_seizures", 0)
                print(f"  {filename} seizures={num_seizures}")
            continue

        for filename in filenames:
            split_dir = output_dir / split_name
            save_path = split_dir / f"{safe_file_stem(patient, filename)}_windows.npz"

            if save_path.exists() and not overwrite:
                try:
                    loaded = np.load(save_path, allow_pickle=True)
                    labels = loaded["y"]
                    summary = {
                        "patient": patient,
                        "filename": filename,
                        "num_windows": int(len(labels)),
                        "num_seizure_windows": int(np.sum(labels == 1)),
                        "num_nonseizure_windows": int(np.sum(labels == 0)),
                        "num_channels": int(loaded["num_channels"]),
                        "save_path": str(save_path),
                        "skipped_existing": True,
                    }
                    processed_summaries.append(summary)

                    if split_name != "test":
                        total_windows += summary["num_windows"]
                        total_seizure += summary["num_seizure_windows"]
                        total_nonseizure += summary["num_nonseizure_windows"]

                    print(f"SKIP processed exists: {save_path}")
                    continue
                except Exception as exc:
                    print(f"WARNING could not read existing {save_path}; reprocessing: {exc}")

            try:
                print(f"Processing {patient}/{filename}")

                specs, labels, window_times, kept_freqs, kept_times, summary = process_one_edf(
                    data_dir,
                    annotations,
                    patient,
                    filename,
                )

                save_path = save_processed_edf(
                    output_dir,
                    split_name,
                    patient,
                    filename,
                    specs,
                    labels,
                    window_times,
                    kept_freqs,
                    kept_times,
                    summary,
                )

                summary = dict(summary)
                summary["save_path"] = str(save_path)
                summary["skipped_existing"] = False
                processed_summaries.append(summary)

                if split_name != "test":
                    total_windows += summary["num_windows"]
                    total_seizure += summary["num_seizure_windows"]
                    total_nonseizure += summary["num_nonseizure_windows"]

                    print(
                        f"  saved={save_path.name}, "
                        f"channels={summary['num_channels']}, "
                        f"windows={summary['num_windows']}, "
                        f"seizure={summary['num_seizure_windows']}, "
                        f"nonseizure={summary['num_nonseizure_windows']}"
                    )
                else:
                    print("  processed and saved separately for final evaluation only")

            except Exception as exc:
                print(f"FAILED {patient}/{filename}: {exc}")
                failed_files.append(
                    {
                        "split": split_name,
                        "patient": patient,
                        "filename": filename,
                        "error": str(exc),
                    }
                )

    summary_path = None

    if not dry_run and processed_summaries:
        summary_path = output_dir / f"{split_name}_processed_files.csv"
        with summary_path.open("w") as f:
            fields = [
                "patient",
                "filename",
                "num_channels",
                "duration_sec",
                "num_windows",
                "num_seizure_windows",
                "num_nonseizure_windows",
                "save_path",
                "skipped_existing",
            ]
            f.write(",".join(fields) + "\n")

            for row in processed_summaries:
                values = [str(row.get(field, "")) for field in fields]
                f.write(",".join(values) + "\n")

    split_summary = {
        "split": split_name,
        "patients": patients,
        "num_patients": len(patients),
        "num_files_processed_or_seen": len(processed_summaries),
        "total_windows": None if split_name == "test" else total_windows,
        "total_seizure_windows": None if split_name == "test" else total_seizure,
        "total_nonseizure_windows": None if split_name == "test" else total_nonseizure,
        "summary_path": str(summary_path) if summary_path else None,
        "dry_run": dry_run,
    }

    return split_summary, failed_files


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_failed_files = []

    for split_name in args.splits:
        patients = resolve_patients_for_split(args, split_name)

        split_summary, failed_files = process_split(
            data_dir,
            output_dir,
            split_name,
            patients,
            include_all_files=args.include_all_files,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

        all_summaries.append(split_summary)
        all_failed_files.extend(failed_files)

    print("\n==============================")
    print("Processing summary")
    print("==============================")

    for summary in all_summaries:
        print(summary)

    if all_failed_files:
        failed_path = output_dir / "failed_files.txt"
        with failed_path.open("w") as f:
            for row in all_failed_files:
                f.write(str(row) + "\n")
        print("Failed files saved to:", failed_path)

    print("\nReminder: test split should remain untouched for tuning/model decisions.")


if __name__ == "__main__":
    main()
