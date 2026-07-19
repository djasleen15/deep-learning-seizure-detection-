"""Process CHB-MIT EDF files into patient-level split datasets.

Important:
- Train patients: chb01-chb18
- Validation patients: chb19-chb21
- Test patients: chb22-chb24

The test split is processed and saved separately only so the final evaluation can
run later. This script does not print test seizure/non-seizure statistics, because
test statistics should not inform model development decisions.
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
        "--include-all-files",
        action="store_true",
        help="Process all EDF files, not only files with seizures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list selected files; do not load/process EDFs or save arrays.",
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


def save_split_arrays(output_dir, split_name, specs_list, labels_list, patient_ids, file_ids):
    if not specs_list:
        print(f"No processed files for split {split_name}; skipping save.")
        return None

    x = np.concatenate(specs_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    patient_ids = np.array(patient_ids)
    file_ids = np.array(file_ids)

    save_path = output_dir / f"{split_name}_windows.npz"

    np.savez_compressed(
        save_path,
        X=x,
        y=y,
        patient_ids=patient_ids,
        file_ids=file_ids,
    )

    return save_path, x, y


def process_split(data_dir, output_dir, split_name, patients, include_all_files, dry_run):
    print(f"\n==============================")
    print(f"Processing split: {split_name}")
    print(f"Patients: {patients}")
    print(f"==============================")

    annotations = load_all_annotations(data_dir, patients)

    specs_list = []
    labels_list = []
    patient_ids = []
    file_ids = []
    failed_files = []
    processed_summaries = []

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
            try:
                print(f"Processing {patient}/{filename}")

                specs, labels, _, _, _, summary = process_one_edf(
                    data_dir,
                    annotations,
                    patient,
                    filename,
                )

                specs_list.append(specs)
                labels_list.append(labels)
                patient_ids.extend([patient] * len(labels))
                file_ids.extend([filename] * len(labels))
                processed_summaries.append(summary)

                if split_name != "test":
                    print(
                        f"  windows={summary['num_windows']}, "
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

    if dry_run:
        split_summary = {
            "split": split_name,
            "patients": patients,
            "num_patients": len(patients),
            "dry_run": True,
        }
        return split_summary, failed_files

    saved = save_split_arrays(
        output_dir,
        split_name,
        specs_list,
        labels_list,
        patient_ids,
        file_ids,
    )

    if saved is None:
        total_windows = 0
        total_seizure = None
        total_nonseizure = None
        save_path = None
    else:
        save_path, _, y = saved
        total_windows = int(len(y))

        if split_name == "test":
            total_seizure = None
            total_nonseizure = None
        else:
            total_seizure = int(np.sum(y == 1))
            total_nonseizure = int(np.sum(y == 0))

    split_summary = {
        "split": split_name,
        "patients": patients,
        "num_patients": len(patients),
        "num_files_processed": len(processed_summaries),
        "total_windows": total_windows,
        "total_seizure_windows": total_seizure,
        "total_nonseizure_windows": total_nonseizure,
        "save_path": str(save_path) if save_path else None,
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
        patients = PATIENT_SPLITS[split_name]

        split_summary, failed_files = process_split(
            data_dir,
            output_dir,
            split_name,
            patients,
            include_all_files=args.include_all_files,
            dry_run=args.dry_run,
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
