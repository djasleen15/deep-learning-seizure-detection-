# EEG Seizure Detection

This repository contains the code used for the APS360 progress-report pipeline for
EEG seizure detection on the CHB-MIT dataset.

## What This Code Covers

- Parse CHB-MIT seizure annotations from patient summary files.
- Load EDF recordings.
- Preprocess EEG using a 0.5-40 Hz bandpass filter and common average reference.
- Segment EEG into non-overlapping 4-second windows.
- Label windows using annotation overlap, including windows that straddle seizure onset.
- Convert windows to log-magnitude STFT spectrograms.
- Build 12-second sequences from three consecutive 4-second spectrograms.
- Run a hand-coded amplitude-threshold baseline.
- Train a lightweight CNN-LSTM proof-of-concept model.
- Evaluate cross-patient generalization without retraining.

## Progress-Report Results From Colab

The report experiments were run in Colab on a small subset:

- Training file: `chb01_03.edf`
- Within-patient validation file: `chb01_04.edf`
- Cross-patient check: `chb02_16.edf`

Expanded data processing was also run on all seizure-containing `chb01` files:

- 7 EDF files
- 5981 total 4-second windows
- 5866 non-seizure windows
- 115 seizure windows

Baseline result on the two-file subset:

- Threshold rule: mean + 2.5 standard deviations
- Minimum channels: 2
- Minimum duration: 2 seconds
- F1: 0.1039
- Sensitivity: 0.5000
- Specificity: 0.9271

Lightweight CNN-LSTM final epoch result:

- F1: 0.3279
- Sensitivity: 1.0000
- Specificity: 0.9538

Cross-patient check on `chb02_16.edf`:

- F1: 0.2840
- Sensitivity: 1.0000
- Specificity: 0.4579

## Setup

```bash
pip install -r requirements.txt
```

If running scripts from the repo root, set:

```bash
export PYTHONPATH="$PWD/src"
```

## Example Commands

Process all seizure-containing `chb01` files:

```bash
python scripts/process_chb01_seizure_files.py \
  --data-dir /content/drive/MyDrive/APS360_seizure_project/chb-mit-data \
  --output-dir processed
```

Train the lightweight CNN-LSTM on the two-file progress-report subset:

```bash
python scripts/train_tiny_cnn_lstm.py \
  --data-dir /content/drive/MyDrive/APS360_seizure_project/chb-mit-data \
  --output-dir models \
  --epochs 5
```

Run the amplitude-threshold baseline grid on `chb01_04.edf`:

```bash
python scripts/run_baseline.py \
  --data-dir /content/drive/MyDrive/APS360_seizure_project/chb-mit-data \
  --patient chb01 \
  --filename chb01_04.edf \
  --output-dir report_figures
```

Evaluate the trained model on unseen patient `chb02_16.edf`:

```bash
python scripts/evaluate_cross_patient.py \
  --data-dir /content/drive/MyDrive/APS360_seizure_project/chb-mit-data \
  --checkpoint models/small_cnn_lstm_chb01_tiny_subset.pt \
  --patient chb02 \
  --filename chb02_16.edf \
  --output-dir report_figures
```

## Notes

The lightweight CNN-LSTM is a progress-report proof of concept. The final project
plan is to scale training across more patients and evaluate strict patient-level
generalization as the main experimental question.
