"""Data loading, annotation parsing, preprocessing, windowing, and STFT utilities."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyedflib
from scipy.signal import butter, sosfiltfilt, stft


def parse_summary_file(summary_path: str | Path) -> dict[str, dict[str, object]]:
    """Parse a CHB-MIT patient summary file into seizure intervals by EDF filename."""
    summary_path = Path(summary_path)
    content = summary_path.read_text()

    file_blocks = content.split("File Name:")[1:]
    parsed: dict[str, dict[str, object]] = {}

    for block in file_blocks:
        lines = block.strip().split("\n")
        filename = lines[0].strip()

        num_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        num_seizures = int(num_match.group(1)) if num_match else 0

        starts = re.findall(
            r"Seizure(?:\s\d+)?\sStart Time:\s*(\d+)\s*seconds",
            block,
        )
        ends = re.findall(
            r"Seizure(?:\s\d+)?\sEnd Time:\s*(\d+)\s*seconds",
            block,
        )
        seizures = [(int(start), int(end)) for start, end in zip(starts, ends)]

        if len(seizures) != num_seizures:
            print(
                f"WARNING: {filename} declares {num_seizures} seizures "
                f"but parsed {len(seizures)}"
            )

        parsed[filename] = {
            "seizures": seizures,
            "num_seizures": num_seizures,
        }

    return parsed


def load_all_annotations(data_dir: str | Path, patients: list[str]) -> dict[str, dict]:
    """Load summary annotations for a list of CHB-MIT patient folders."""
    data_dir = Path(data_dir)
    annotations = {}

    for patient in patients:
        summary_path = data_dir / patient / f"{patient}-summary.txt"
        annotations[patient] = parse_summary_file(summary_path)

    return annotations


def load_edf(edf_path: str | Path) -> dict[str, object]:
    """Load an EDF file as channels x samples."""
    edf_path = Path(edf_path)
    reader = pyedflib.EdfReader(str(edf_path))

    try:
        n_channels = reader.signals_in_file
        channel_labels = reader.getSignalLabels()
        fs_values = reader.getSampleFrequencies()

        if len(set(fs_values)) != 1:
            raise ValueError(f"Expected one sampling rate, got: {fs_values}")

        signals = np.vstack([reader.readSignal(i) for i in range(n_channels)])
        fs = float(fs_values[0])
        duration_sec = signals.shape[1] / fs

        return {
            "signals": signals,
            "fs": fs,
            "n_channels": n_channels,
            "channel_labels": channel_labels,
            "duration_sec": duration_sec,
        }
    finally:
        reader.close()


def bandpass_filter(
    signals: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a numerically stable SOS bandpass filter."""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signals, axis=1)


def common_average_reference(signals: np.ndarray) -> np.ndarray:
    """Apply common average referencing across channels."""
    avg = np.mean(signals, axis=0, keepdims=True)
    return signals - avg


def preprocess_signal(raw_signals: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass filter and re-reference raw EEG signals."""
    filtered = bandpass_filter(raw_signals, fs)
    return common_average_reference(filtered)


def label_window(
    window_start_sec: float,
    window_end_sec: float,
    seizure_intervals: list[tuple[int, int]],
    min_overlap_sec: float = 1.0,
) -> int:
    """Label a window as seizure if it overlaps annotations enough."""
    for seizure_start, seizure_end in seizure_intervals:
        overlap_start = max(window_start_sec, seizure_start)
        overlap_end = min(window_end_sec, seizure_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap >= min_overlap_sec:
            return 1

    return 0


def create_windows(
    signals: np.ndarray,
    fs: float,
    seizure_intervals: list[tuple[int, int]],
    window_sec: int = 4,
    stride_sec: int = 4,
    min_overlap_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Segment EEG signals into fixed-length windows and labels."""
    window_samples = int(window_sec * fs)
    stride_samples = int(stride_sec * fs)
    total_samples = signals.shape[1]

    windows = []
    labels = []
    window_times = []

    for start_sample in range(0, total_samples - window_samples + 1, stride_samples):
        end_sample = start_sample + window_samples
        start_sec = start_sample / fs
        end_sec = end_sample / fs

        windows.append(signals[:, start_sample:end_sample])
        labels.append(
            label_window(
                start_sec,
                end_sec,
                seizure_intervals,
                min_overlap_sec=min_overlap_sec,
            )
        )
        window_times.append((start_sec, end_sec))

    return np.array(windows), np.array(labels), window_times


def window_to_spectrogram(
    window: np.ndarray,
    fs: float,
    nperseg: int = 128,
    noverlap: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one channels x samples EEG window to log-magnitude STFT."""
    freqs, times, zxx = stft(
        window,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=1,
    )
    return np.log1p(np.abs(zxx)), freqs, times


def windows_to_spectrograms(
    windows: np.ndarray,
    fs: float,
    max_freq: float = 40,
    nperseg: int = 128,
    noverlap: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert all windows to log-magnitude STFT spectrograms."""
    all_specs = []
    kept_freqs = None
    kept_times = None

    for window in windows:
        spec, freqs, times = window_to_spectrogram(
            window,
            fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        freq_mask = freqs <= max_freq
        all_specs.append(spec[:, freq_mask, :])
        kept_freqs = freqs[freq_mask]
        kept_times = times

    return np.array(all_specs), kept_freqs, kept_times


def process_one_edf(
    data_dir: str | Path,
    annotations: dict[str, dict],
    patient: str,
    filename: str,
    window_sec: int = 4,
    stride_sec: int = 4,
    min_overlap_sec: float = 1.0,
    max_freq: float = 40,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], np.ndarray, np.ndarray, dict]:
    """Run the full preprocessing/window/STFT pipeline for one EDF file."""
    data_dir = Path(data_dir)
    edf_data = load_edf(data_dir / patient / filename)
    processed = preprocess_signal(edf_data["signals"], edf_data["fs"])
    seizure_intervals = annotations[patient][filename]["seizures"]

    windows, labels, window_times = create_windows(
        processed,
        edf_data["fs"],
        seizure_intervals,
        window_sec=window_sec,
        stride_sec=stride_sec,
        min_overlap_sec=min_overlap_sec,
    )
    specs, kept_freqs, kept_times = windows_to_spectrograms(
        windows,
        edf_data["fs"],
        max_freq=max_freq,
    )

    summary = {
        "patient": patient,
        "filename": filename,
        "fs": edf_data["fs"],
        "num_channels": edf_data["n_channels"],
        "duration_sec": edf_data["duration_sec"],
        "num_windows": specs.shape[0],
        "num_nonseizure_windows": int(np.sum(labels == 0)),
        "num_seizure_windows": int(np.sum(labels == 1)),
        "spectrogram_shape_per_window": specs.shape[1:],
        "seizure_intervals": seizure_intervals,
    }

    return specs, labels, window_times, kept_freqs, kept_times, summary


def create_spectrogram_sequences(
    x: np.ndarray,
    y: np.ndarray,
    file_ids: np.ndarray,
    seq_len: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group consecutive 4-second spectrogram windows into 12-second sequences."""
    x_seq = []
    y_seq = []
    seq_file_ids = []
    file_ids = np.array(file_ids)

    for filename in np.unique(file_ids):
        mask = file_ids == filename
        x_file = x[mask]
        y_file = y[mask]

        for i in range(0, len(y_file) - seq_len + 1):
            x_seq.append(x_file[i : i + seq_len])
            y_seq.append(1 if np.any(y_file[i : i + seq_len] == 1) else 0)
            seq_file_ids.append(filename)

    return np.array(x_seq), np.array(y_seq), np.array(seq_file_ids)


def reshape_sequences_for_cnn(x_seq: np.ndarray) -> np.ndarray:
    """Reshape sequences for the lightweight CNN-LSTM."""
    n, seq_len, eeg_channels, freq_bins, time_bins = x_seq.shape
    return x_seq.reshape(n, seq_len, 1, eeg_channels * freq_bins, time_bins).astype(
        np.float32
    )

