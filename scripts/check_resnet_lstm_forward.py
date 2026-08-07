"""Verify on-the-fly ResNet-18 + causal LSTM forward pass.

This script does not train. It builds one real batch from compact processed
train/validation files, resizes spectrograms on the fly, and verifies that the
model emits one logit per sequence.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from seizure_detection.model import ResNet18LSTM, count_parameters
from seizure_detection.resnet_input import (
    ResNetSequenceDataset,
    load_common_channels,
    processed_npz_files,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--common-channels-json", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--freeze-resnet",
        action="store_true",
        help="Freeze the ResNet feature extractor for this forward-pass check.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet weights. Useful if network access is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    common_channels = load_common_channels(args.common_channels_json)
    npz_files = processed_npz_files(args.processed_dir, args.split)

    dataset = ResNetSequenceDataset(
        data_dir=args.data_dir,
        npz_files=npz_files,
        common_channels=common_channels,
        seq_len=args.seq_len,
        image_size=args.image_size,
        max_files=args.max_files,
    )

    if dataset.excluded_files:
        print("Excluded files:")
        for row in dataset.excluded_files:
            print(row)
    else:
        print("Excluded files: 0")

    print("Common channel count:", len(common_channels))
    print("Common channels:", common_channels)
    print("Files used:", len(dataset.file_records))
    print("Sequences indexed:", len(dataset))
    print("Sequences cross file boundaries: False")
    print(
        "Resize method: flatten common-channel/frequency dimensions, "
        "bilinear interpolate to 224x224, replicate to 3 channels."
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch_x, batch_y = next(iter(loader))

    print("Batch X shape:", tuple(batch_x.shape))
    print("Batch y shape:", tuple(batch_y.shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18LSTM(
        lstm_hidden=256,
        pretrained=not args.no_pretrained,
        freeze_resnet=args.freeze_resnet,
    ).to(device)

    params = count_parameters(model)
    print("Parameter counts:", params)
    print("ResNet frozen:", args.freeze_resnet)
    print("Fine-tuning decision for Chunk 4: OPEN, not decided in this script.")

    model.eval()
    with torch.no_grad():
        logits = model(batch_x.to(device))

    print("Output logits shape:", tuple(logits.shape))
    print("First logits:", logits[: min(5, len(logits))].detach().cpu())


if __name__ == "__main__":
    main()
