"""Restart-safe direct downloader for the CHB-MIT dataset.

This intentionally uses the PhysioNet RECORDS manifest and direct file URLs
instead of recursive wget. Downloads are explicit, resumable at the file level,
and easier to run in Colab where long jobs can be interrupted.
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


PHYSIONET_CHBMIT_BASE_URL = "https://physionet.org/files/chbmit/1.0.0"


def patient_id(patient_number: int) -> str:
    """Format a CHB-MIT patient number as chbXX."""
    return f"chb{patient_number:02d}"


def fetch_records_manifest(base_url: str = PHYSIONET_CHBMIT_BASE_URL) -> list[str]:
    """Fetch the PhysioNet RECORDS manifest as relative EDF paths."""
    manifest_url = f"{base_url}/RECORDS"
    with urllib.request.urlopen(manifest_url, timeout=60) as response:
        content = response.read().decode("utf-8")

    return [
        line.strip()
        for line in content.splitlines()
        if line.strip() and line.strip().endswith(".edf")
    ]


def records_for_patients(
    patients: list[str],
    base_url: str = PHYSIONET_CHBMIT_BASE_URL,
) -> dict[str, list[str]]:
    """Return EDF manifest entries grouped by patient."""
    records = fetch_records_manifest(base_url=base_url)
    grouped = {patient: [] for patient in patients}

    for record in records:
        patient = record.split("/", 1)[0]
        if patient in grouped:
            grouped[patient].append(record)

    return grouped


def _download_url(
    url: str,
    output_path: Path,
    timeout: int = 120,
    retries: int = 3,
    sleep_sec: int = 5,
) -> bool:
    """Download one URL with skip-if-exists and wget resume support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"SKIP exists: {output_path}")
        return False

    part_path = output_path.with_suffix(output_path.suffix + ".part")

    # If an earlier urllib run left a .part file, let wget resume from it.
    # wget resumes only when the output filename is the partial filename.
    wget_output_path = part_path if part_path.exists() else output_path

    for attempt in range(1, retries + 1):
        print(f"DOWNLOAD attempt {attempt}/{retries}: {url}")

        cmd = [
            "wget",
            "-c",
            "--timeout",
            str(timeout),
            "--tries",
            "1",
            "-O",
            str(wget_output_path),
            url,
        ]

        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            if wget_output_path == part_path:
                part_path.replace(output_path)

            print(f"SAVED: {output_path}")
            return True

        print(f"WARNING failed attempt {attempt} for {url}: wget exit code {result.returncode}")

        if attempt < retries:
            time.sleep(sleep_sec)

    raise RuntimeError(f"Failed to download after {retries} attempts: {url}")


def download_patient(
    patient: str,
    output_dir: str | Path,
    base_url: str = PHYSIONET_CHBMIT_BASE_URL,
    timeout: int = 120,
    retries: int = 3,
) -> dict[str, int]:
    """Download one patient's summary file and all EDFs listed in RECORDS."""
    output_dir = Path(output_dir)
    grouped_records = records_for_patients([patient], base_url=base_url)
    records = grouped_records[patient]

    if not records:
        raise ValueError(f"No EDF records found in PhysioNet manifest for {patient}")

    downloaded = 0
    skipped = 0

    summary_rel = f"{patient}/{patient}-summary.txt"
    summary_saved = _download_url(
        f"{base_url}/{summary_rel}",
        output_dir / summary_rel,
        timeout=timeout,
        retries=retries,
    )
    downloaded += int(summary_saved)
    skipped += int(not summary_saved)

    for record in records:
        saved = _download_url(
            f"{base_url}/{record}",
            output_dir / record,
            timeout=timeout,
            retries=retries,
        )
        downloaded += int(saved)
        skipped += int(not saved)

    return {
        "patient": patient,
        "records": len(records),
        "downloaded": downloaded,
        "skipped": skipped,
    }


def download_patients(
    patients: list[str],
    output_dir: str | Path,
    base_url: str = PHYSIONET_CHBMIT_BASE_URL,
    timeout: int = 120,
    retries: int = 3,
) -> list[dict[str, int]]:
    """Download multiple patients, preserving progress patient by patient."""
    results = []

    for patient in patients:
        print(f"\n=== {patient} ===")
        result = download_patient(
            patient,
            output_dir=output_dir,
            base_url=base_url,
            timeout=timeout,
            retries=retries,
        )
        print(result)
        results.append(result)

    return results
