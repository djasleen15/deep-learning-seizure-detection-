# Automated Seizure Detection from EEG

A patient-level, leakage-aware seizure detection pipeline built on the
CHB-MIT Scalp EEG Database, developed as an APS360 course project at the
University of Toronto and as a portfolio piece for research applications.

## Overview

This project builds an automated seizure detection system that classifies
12-second EEG sequences, made from three consecutive 4-second windows, as
seizure or non-seizure. It also includes Grad-CAM interpretability to highlight
the time-frequency regions that contributed to a prediction.

**Core contribution:** rather than proposing a novel architecture, this project
evaluates how detection performance scales with the number of training patients
under a strict patient-level train/validation/test split. This is motivated by
evidence that much prior work on this dataset does not rigorously separate
patients between training and evaluation, risking inflated performance estimates
from data leakage.

## Key Results

| Model | Split | Sensitivity | Specificity | Precision | F1 |
|---|---|---:|---:|---:|---:|
| Baseline amplitude threshold | Validation | 0.189 | 0.999 | 0.712 | 0.298 |
| CNN-LSTM, 6 train patients | Validation | 0.403 | 0.994 | 0.595 | 0.480 |
| CNN-LSTM, 6 train patients | Held-out test | 0.371 | 0.979 | 0.230 | 0.284 |

The drop from validation F1 to held-out test F1 is a central result: performance
measured before the final unseen-patient evaluation overstated generalization.

Patient-count scaling on the fixed validation set was non-monotonic:

| Train patients | Validation F1 |
|---:|---:|
| 3 | 0.266 |
| 6 | 0.480 |
| 12 | 0.420 |

## Data

This project uses the
[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/), a
PhysioNet pediatric scalp EEG dataset sampled at 256 Hz.

The split was locked at the patient level:

- Train: `chb01`-`chb18`
- Validation: `chb19`-`chb21`
- Held-out test: `chb22`-`chb24`

Raw EDF files and processed NumPy arrays are not included in this repository.

## Pipeline

1. **Preprocessing:** 0.5-40 Hz bandpass filtering, common average reference,
   non-overlapping 4-second windows, and seizure labels assigned by at least
   1 second of overlap with an annotated seizure interval.
2. **Channel standardization:** 22 channels selected using train/validation
   metadata only, with test files mapped to this frozen list.
3. **Spectrogram conversion:** log-magnitude STFT with 0-40 Hz retained.
4. **Sequence construction:** sliding 3-window sequences, producing one
   12-second sequence-level label.
5. **Baseline:** hand-coded amplitude-threshold detector with no learned
   parameters.
6. **Primary model:** lightweight CNN-LSTM with 2 convolutional layers, a
   128-dimensional CNN feature vector, a causal unidirectional LSTM with hidden
   size 64, and a binary classifier.
7. **Interpretability:** Grad-CAM applied to the final convolutional layer.

The originally proposed ResNet-18 + LSTM architecture was implemented and
forward-pass verified, but full fine-tuning was infeasible under available
free-tier Colab runtime constraints. The final report describes this
infrastructure limitation and the resulting lightweight CNN-LSTM choice.

## Repository Structure

- `src/seizure_detection/` - reusable pipeline modules
- `scripts/` - command-line entry points for download, processing, training,
  evaluation, and visualization
- `report/` - progress/final reports and final report figures
- `results/` - compact summary metrics used in the report
- `notebooks/` - optional exploratory notebooks

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Example Commands

Download CHB-MIT patients:

```bash
python scripts/download_chbmit_patients.py \
  --output-dir data/chb-mit-data \
  --start 1 \
  --end 24
```

Process patient-level splits:

```bash
python scripts/process_patient_splits.py \
  --data-dir data/chb-mit-data \
  --output-dir data/chb-mit-data/processed_patient_splits \
  --splits train val
```

Run the full-scale baseline:

```bash
python scripts/run_full_baseline.py \
  --data-dir data/chb-mit-data \
  --processed-dir data/chb-mit-data/processed_patient_splits \
  --output-dir results
```

Train the lightweight CNN-LSTM:

```bash
python scripts/train_lightweight_cnn_lstm_full.py \
  --data-dir data/chb-mit-data \
  --processed-dir data/chb-mit-data/processed_patient_splits \
  --common-channels-json data/chb-mit-data/model_input_outputs/common_train_val_channels.json \
  --channel-labels-csv data/chb-mit-data/model_input_outputs/train_val_channel_labels.csv \
  --output-dir models \
  --train-patients chb01 chb02 chb03 chb04 chb05 chb06 \
  --epochs 4
```

## Reports

- [Final Report](report/final_report.pdf)
- [Progress Report](report/progress_report.pdf)

## Author

Jasleen Dhaliwal - University of Toronto, Computer Engineering
