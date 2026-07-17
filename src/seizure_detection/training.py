"""PyTorch datasets, training loops, and evaluation helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from seizure_detection.baseline import compute_binary_metrics_clean


class EEGSequenceDataset(Dataset):
    """Dataset for spectrogram sequences and binary labels."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def train_one_epoch(model, train_loader, criterion, optimizer, device) -> float:
    """Train for one epoch and return mean loss."""
    model.train()
    total_loss = 0
    total_examples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_examples += batch_x.size(0)

    return total_loss / total_examples


def evaluate_model(model, data_loader, criterion, device, threshold: float = 0.5):
    """Evaluate a model and return loss, metrics, probabilities, predictions, labels."""
    model.eval()
    total_loss = 0
    total_examples = 0
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            total_loss += loss.item() * batch_x.size(0)
            total_examples += batch_x.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / total_examples
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    metrics = compute_binary_metrics_clean(all_labels, all_preds)

    return avg_loss, metrics, all_probs, all_preds, all_labels

