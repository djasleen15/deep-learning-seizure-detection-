"""Evaluate a trained lightweight CNN-LSTM on an unseen patient EDF file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seizure_detection.data_processing import (
    create_spectrogram_sequences,
    load_all_annotations,
    process_one_edf,
    reshape_sequences_for_cnn,
)
from seizure_detection.model import SmallCNNLSTM
from seizure_detection.training import EEGSequenceDataset, evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient", default="chb02")
    parser.add_argument("--filename", default="chb02_16.edf")
    parser.add_argument("--output-dir", default="report_figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_all_annotations(data_dir, [args.patient])
    specs, labels, *_ = process_one_edf(
        data_dir,
        annotations,
        args.patient,
        args.filename,
    )
    file_ids = np.array([args.filename] * len(labels))
    x_seq, y_seq, _ = create_spectrogram_sequences(specs, labels, file_ids)
    x_model = reshape_sequences_for_cnn(x_seq)
    y_model = y_seq.astype(np.float32)

    dataset = EEGSequenceDataset(x_model, y_model)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNNLSTM().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    loss, metrics, probs, preds, true_labels = evaluate_model(
        model,
        loader,
        criterion,
        device,
    )

    result = {
        "patient": args.patient,
        "filename": args.filename,
        "loss": loss,
        "predicted_seizure_sequences": int(np.sum(preds == 1)),
        "true_seizure_sequences": int(np.sum(true_labels == 1)),
        **metrics,
    }
    print(result)
    pd.DataFrame([result]).to_csv(
        output_dir / f"{args.patient}_{args.filename}_cross_patient_metrics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()

