"""Download CHB-MIT patients using direct PhysioNet manifest URLs.

Examples:
  python scripts/download_chbmit_patients.py --output-dir data/chb-mit-data --patients chb03 chb04
  python scripts/download_chbmit_patients.py --output-dir data/chb-mit-data --start 3 --end 24
  python scripts/download_chbmit_patients.py --output-dir data/chb-mit-data --patients chb03 --manifest-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from seizure_detection.download import (
    download_patients,
    patient_id,
    records_for_patients,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Destination chb-mit-data folder")
    parser.add_argument("--patients", nargs="*", help="Explicit patients, e.g. chb03 chb04")
    parser.add_argument("--start", type=int, default=3, help="First patient number")
    parser.add_argument("--end", type=int, default=24, help="Last patient number, inclusive")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only list matching EDF records without downloading files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patients = args.patients or [patient_id(i) for i in range(args.start, args.end + 1)]

    print("Patients to download:", patients)
    print("Output directory:", output_dir)

    if args.manifest_only:
        grouped = records_for_patients(patients)
        for patient, records in grouped.items():
            print(f"{patient}: {len(records)} EDF files")
            for record in records:
                print(f"  {record}")
        return

    results = download_patients(
        patients,
        output_dir=output_dir,
        timeout=args.timeout,
        retries=args.retries,
    )

    results_path = output_dir / "download_results.csv"

    with results_path.open("w") as f:
        f.write("patient,records,downloaded,skipped\n")
        for row in results:
            f.write(
                f"{row['patient']},{row['records']},"
                f"{row['downloaded']},{row['skipped']}\n"
            )

    print("\nDownload summary:")
    for row in results:
        print(row)
    print("Saved:", results_path)


if __name__ == "__main__":
    main()
