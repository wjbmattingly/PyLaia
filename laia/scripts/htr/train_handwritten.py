#!/usr/bin/env python3
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import os
from typing import Dict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from laia.data.handwritten_dataset import HandwrittenDataset
from laia.engine.engine_module import EngineModule
from laia.losses.ctc_loss import CTCLoss
from laia.models.htr.laia_crnn import LaiaCRNN
from laia.utils.symbols_table import SymbolsTable
from laia.common.arguments import OptimizerArgs, SchedulerArgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory containing images")
    parser.add_argument("--train_file", type=str, required=True, help="Training data JSON file")
    parser.add_argument("--val_file", type=str, required=True, help="Validation data JSON file")
    parser.add_argument("--char_map", type=str, required=True, help="Character map JSON file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID (None for CPU)")
    return parser.parse_args()


def load_char_map(char_map_file: str) -> Dict[str, int]:
    with open(char_map_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load character map
    char_map = load_char_map(args.char_map)
    num_symbols = len(char_map)

    # Create datasets
    train_dataset = HandwrittenDataset(
        data_file=args.train_file,
        char_map=char_map,
        img_dir=args.data_dir
    )
    val_dataset = HandwrittenDataset(
        data_file=args.val_file,
        char_map=char_map,
        img_dir=args.data_dir
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=HandwrittenDataset.collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=HandwrittenDataset.collate_fn,
        pin_memory=True
    )

    # Create model
    model = LaiaCRNN(
        num_input_channels=1,
        num_output_labels=num_symbols,
        cnn_num_features=[16, 32, 48, 64],
        cnn_kernel_size=[(3, 3)] * 4,
        cnn_stride=[(1, 1)] * 4,
        cnn_dilation=[(1, 1)] * 4,
        cnn_activation=[torch.nn.ReLU] * 4,
        cnn_poolsize=[(2, 2), (2, 2), (2, 2), (0, 0)],
        cnn_dropout=[0.2] * 4,
        cnn_batchnorm=[True] * 4,
        image_sequencer="maxpool-8",
        rnn_units=256,
        rnn_layers=3,
        rnn_dropout=0.5,
        lin_dropout=0.5
    )

    # Create criterion
    criterion = CTCLoss(reduction="mean")

    # Create optimizer args
    optimizer = OptimizerArgs(
        name=OptimizerArgs.Name.Adam,
        learning_rate=args.learning_rate
    )

    # Create scheduler args
    scheduler = SchedulerArgs(
        active=True,
        monitor="va_loss",
        patience=5,
        factor=0.5
    )

    # Create engine
    engine = EngineModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpu is not None else "cpu",
        devices=[args.gpu] if args.gpu is not None else None,
        default_root_dir=args.output_dir,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="va_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch:02d}-{va_loss:.4f}"
            ),
            pl.callbacks.EarlyStopping(
                monitor="va_loss",
                mode="min",
                patience=10
            )
        ]
    )

    # Train
    trainer.fit(engine, train_loader, val_loader)


if __name__ == "__main__":
    main() 