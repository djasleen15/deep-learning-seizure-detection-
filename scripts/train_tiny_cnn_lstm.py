"""Train the lightweight CNN-LSTM on the two-file chb01 progress-report subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from seizure_detection.data_processing import (
    create_spectrogram_sequences,
    load_all_annotations,
    process_one_edf,
    reshape_sequences_for_cnn,
)
from seizure_detection.model import SmallCNNLSTM
from seizure_detection.training import EEGSequenceDataset, evaluate_model, train_one_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to chb-mit-data folder")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_all_annotations(data_dir, ["chb01"])
    subset_files = ["chb01_03.edf", "chb01_04.edf"]

    specs_list = []
    labels_list = []
    file_ids = []

    for filename in subset_files:
        specs, labels, *_ = process_one_edf(data_dir, annotations, "chb01", filename)
        specs_list.append(specs)
        labels_list.append(labels)
        file_ids.extend([filename] * len(labels))

    x = np.concatenate(specs_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    file_ids = np.array(file_ids)
    x_seq, y_seq, seq_file_ids = create_spectrogram_sequences(x, y, file_ids)
    x_model = reshape_sequences_for_cnn(x_seq)
    y_model = y_seq.astype(np.float32)

    train_mask = seq_file_ids == "chb01_03.edf"
    val_mask = seq_file_ids == "chb01_04.edf"
    x_train, y_train = x_model[train_mask], y_model[train_mask]
    x_val, y_val = x_model[val_mask], y_model[val_mask]

    train_dataset = EEGSequenceDataset(x_train, y_train)
    val_dataset = EEGSequenceDataset(x_val, y_val)

    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNNLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, *_ = evaluate_model(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_sensitivity": val_metrics["sensitivity_recall"],
            "val_specificity": val_metrics["specificity"],
        }
        history.append(row)
        print(row)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "small_cnn_lstm_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "input_shape": x_model.shape[1:],
        },
        output_dir / "small_cnn_lstm_chb01_tiny_subset.pt",
    )
    print("Saved model and history to", output_dir)


if __name__ == "__main__":
    main()

