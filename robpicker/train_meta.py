"""
Meta-learning training script for cryo-ET particle picking.

This script trains a segmentation model using meta-learning to:
- Learn class weights for loss reweighting
- Learn label corrections for noisy/imprecise annotations
- Learn per-voxel loss weights
- Learn voxel-wise mixup coefficients (optional)

Usage:
    python train_meta.py -C cfg_resnet34_meta 

Arguments:
    -C, --config: Config file name (without .py extension)
    --resume: Path to checkpoint file to resume training from
"""

import numpy as np
import pandas as pd
import importlib
import sys
import os
import argparse
import logging
import shutil
from copy import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from betty.configs import Config, EngineConfig

from robpicker.meta.meta_modules import ClassWeightReweight, FeatureMapCorrect, LabelCorrect, LossWeightModule, MetaMixupModule
from robpicker.meta.problems import MainTask, Reweight, Correct, LossWeight, MetaMixup
from robpicker.meta.engine import MetaEngine

from robpicker.utils import (
    set_seed,
    get_data,
    worker_init_fn,
    get_cosine_schedule_with_warmup,
    load_config,
)
from robpicker.data import ds


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logging(output_dir):
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_model(cfg):
    """Get the meta-learning enabled model."""
    Net = importlib.import_module(cfg.model).Net
    net = Net(cfg)
    return net


def build_scheduler(cfg, optimizer, total_steps):
    """Build LR scheduler for meta training based on cfg.schedule."""
    if cfg.schedule != "cosine":
        return None
    warmup_steps = int(getattr(cfg, "warmup_steps", 0))
    warmup_steps = max(0, min(warmup_steps, total_steps))
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=getattr(cfg, "num_cycles", 0.5),
    )


def log_train_sample_frequency(train_loader, cfg, logger):
    """Log per-class sample presence frequency from the training loader."""
    max_batches = int(getattr(cfg, "resample_stats_batches", 0))
    if max_batches <= 0:
        return

    total_samples = 0
    class_counts = torch.zeros(len(cfg.classes), dtype=torch.long)
    bg_only = 0

    loader_iter = iter(train_loader)
    for _ in range(min(max_batches, len(train_loader))):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        targets = batch.get("target", None)
        if targets is None:
            continue
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)

        b, c = targets.shape[:2]
        present = targets.view(b, c, -1).sum(dim=2) > 0
        class_counts += present.sum(dim=0).cpu()
        bg_only += int((present.sum(dim=1) == 0).sum().item())
        total_samples += b

    if total_samples == 0:
        logger.info("Train sample frequency: no samples counted")
        return

    freqs = (class_counts.float() / float(total_samples)).tolist()
    parts = [f"{cls}={freq:.3f}" for cls, freq in zip(cfg.classes, freqs)]
    logger.info(
        "Train sample frequency over %d samples (%d batches): %s; bg_only=%.3f",
        total_samples,
        min(max_batches, len(train_loader)),
        ", ".join(parts),
        bg_only / float(total_samples),
    )


def get_meta_dataloaders(train_df, meta_df, val_df, cfg):
    """
    Create dataloaders for meta-learning.

    Returns train loader (for main task) and meta loader (for meta modules).

    Supports two modes:
    1. Uses EMPIAR format data with pre-split train/meta/test folders
    2. Uses train_df and meta_df provided by config
    """
    dataset_type = getattr(cfg, 'dataset', 'robpicker.data.ds')

    prefetch_factor = int(getattr(cfg, "prefetch_factor", 4))
    use_persistent_workers = bool(getattr(cfg, "persistent_workers", True))

    def _loader_kwargs():
        kwargs = {
            "pin_memory": True,
            "prefetch_factor": max(prefetch_factor, 1),
            "persistent_workers": use_persistent_workers,
        }
        if cfg.num_workers == 0:
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("persistent_workers", None)
        return kwargs

    if dataset_type in ('robpicker.data.ds', 'ds'):
        # Default mode: Use ds.CustomDataset with pre-split folders
        print(f"Using dataset format for training: {cfg.data_dir}")

        # Create training dataset
        train_dataset = ds.CustomDataset(None, cfg, aug=cfg.train_aug, mode="train")
        train_collate_fn = ds.tr_collate_fn

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=train_collate_fn,
            drop_last=cfg.drop_last,
            worker_init_fn=worker_init_fn,
            **_loader_kwargs(),
        )

        # Create meta dataset from meta folder
        meta_dataset = ds.CustomDataset(None, cfg, aug=cfg.train_aug, mode="meta")
        meta_collate = ds.tr_collate_fn

        meta_loader = DataLoader(
            meta_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=meta_collate,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            **_loader_kwargs(),
        )

        print(f"Training: {len(train_dataset.tomograms)} tomograms, {len(train_dataset.df)} annotations")
        print(f"Meta: {len(meta_dataset.tomograms)} tomograms, {len(meta_dataset.df)} annotations")

        # Validation dataset from test folder (fallback to meta if empty)
        val_dataset = ds.CustomDataset(None, cfg, aug=cfg.val_aug, mode="val")
        val_collate_fn = ds.val_collate_fn
        if len(val_dataset.tomograms) == 0:
            print("No EMPIAR validation tomograms found; using meta dataset for validation.")
            val_dataset = meta_dataset
            val_collate_fn = meta_collate

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size_val if cfg.batch_size_val else cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=val_collate_fn,
            worker_init_fn=worker_init_fn,
            **_loader_kwargs(),
        )
        print(f"Validation: {len(val_dataset.tomograms)} tomograms, {len(val_dataset.df)} annotations")

        return train_loader, meta_loader, val_loader

    else:
        if meta_df is None or len(meta_df) == 0:
            raise ValueError("Meta-learning requires a non-empty meta_df.")

        print(f"Main training: {len(train_df)} samples")
        print(f"Meta dataset: {len(meta_df)} samples")

        train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=cfg.tr_collate_fn,
            drop_last=cfg.drop_last,
            worker_init_fn=worker_init_fn,
            **_loader_kwargs(),
        )

        meta_dataset = cfg.CustomDataset(meta_df, cfg, aug=cfg.train_aug, mode="train")
        meta_loader = DataLoader(
            meta_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=cfg.tr_collate_fn,
            drop_last=cfg.drop_last,
            worker_init_fn=worker_init_fn,
            **_loader_kwargs(),
        )

        # Validation dataloader (for evaluation)
        if val_df is None or len(val_df) == 0:
            print("No validation set provided; using meta dataset for validation.")
            val_dataset = meta_dataset
            val_collate_fn = cfg.tr_collate_fn
        else:
            val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")
            val_collate_fn = cfg.val_collate_fn

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size_val if cfg.batch_size_val else cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=val_collate_fn,
            worker_init_fn=worker_init_fn,
            **_loader_kwargs(),
        )

        return train_loader, meta_loader, val_loader


def train_meta(cfg, args, config_path):
    """Main meta-learning training function."""
    # Set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    set_seed(cfg.seed)

    # Setup device
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    # Setup output directory
    if getattr(cfg, 'fold', None) is not None:
        cfg.output_dir = os.path.join(cfg.output_dir, f"fold{cfg.fold}")
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # save config to output dir
    shutil.copyfile(config_path, os.path.join(output_dir, os.path.basename(config_path)))

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting meta-learning training with config: {cfg.name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Arguments: " + str(args))
    logger.info("Config: " + str(cfg))
    logger.info(f"Reweight: {cfg.reweight}, Correct: {cfg.correct}, LossWeight: {cfg.loss_weight}, MetaMixup: {cfg.meta_mixup}")
    logger.info(f"Unroll steps: {cfg.unroll_steps}")
    logger.info(f"Device: {device}")

    # Get data
    if cfg.dataset != "robpicker.data.ds":
        train_df, meta_df, val_df = get_data(cfg)
        logger.info(
            "Train samples: %d, Meta samples: %d, Val samples: %d",
            len(train_df),
            len(meta_df) if meta_df is not None else 0,
            len(val_df) if val_df is not None else 0,
        )
    else:
        train_df, meta_df, val_df = None, None, None  # EMPIAR dataset handles data internally

    # Get dataloaders
    train_loader, meta_loader, val_loader = get_meta_dataloaders(train_df, meta_df, val_df, cfg)
    logger.info(f"Train batches: {len(train_loader)}, Meta batches: {len(meta_loader)}, Val batches: {len(val_loader)}")
    log_train_sample_frequency(meta_loader, cfg, logger)

    # Create main model
    main_model = get_model(cfg)
    main_model.to(device)
    logger.info(f"Main model created with {sum(p.numel() for p in main_model.parameters())} parameters")

    # Setup Betty configs
    main_config = Config(
        type=cfg.meta_type,
        log_step=cfg.log_step,
        unroll_steps=cfg.unroll_steps,
        warmup_steps=cfg.warmup_steps,
        allow_unused=True,
    )
    meta_config = Config(
        type=cfg.meta_type,
        log_step=cfg.log_step,
        retain_graph=True,
        allow_unused=True,
    )
    engine_config = EngineConfig(
        train_iters=cfg.unroll_steps * cfg.train_iters + cfg.warmup_steps,
        valid_step=cfg.valid_steps,
        roll_back=cfg.rollback,
        logger_type="tensorboard",
    )
    total_steps = engine_config.train_iters
    cfg.total_steps = total_steps
    logger.info(f"Total training steps: {total_steps}")

    # Setup main optimizer
    main_optimizer = optim.Adam(
        main_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    main_scheduler = build_scheduler(cfg, main_optimizer, total_steps)

    # Create MainTask problem
    maintask = MainTask(
        name="main",
        module=main_model,
        optimizer=main_optimizer,
        scheduler=main_scheduler,
        train_data_loader=train_loader,
        config=main_config,
        device=device,
        external_config=cfg,
    )

    problems = [maintask]
    l2u = {maintask: []}
    u2l = {}

    # Setup Reweight problem if enabled
    reweight = None
    if cfg.reweight:
        # Initialize with default class weights if available
        init_weights = None
        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            init_weights = cfg.class_weights

        reweight_model = ClassWeightReweight(
            num_classes=cfg.n_classes + 1,  # +1 for background
            init_weights=init_weights,
        )
        reweight_model.to(device)
        logger.info(f"Reweight module parameters: {sum(p.numel() for p in reweight_model.parameters())}")

        reweight_optimizer = optim.Adam(
            reweight_model.parameters(),
            lr=cfg.meta_lr,
            weight_decay=cfg.meta_weight_decay,
        )
        reweight_scheduler = build_scheduler(cfg, reweight_optimizer, cfg.train_iters)

        reweight = Reweight(
            name="reweight",
            module=reweight_model,
            optimizer=reweight_optimizer,
            scheduler=reweight_scheduler,
            train_data_loader=meta_loader,
            config=meta_config,
            device=device,
            external_config=cfg,
        )

        problems.append(reweight)
        l2u[maintask].append(reweight)
        u2l[reweight] = [maintask]
        logger.info("Reweight module created")

    # Setup Correct problem if enabled
    correct = None
    if cfg.correct:
        correct_type = getattr(cfg, 'correct_type', 'simple')

        if correct_type == 'simple':
            # LabelCorrect: simple 1x1 conv on labels only
            correct_model = LabelCorrect(
                num_classes=cfg.n_classes + 1,  # +1 for background
                hidden_channels=16,
                temperature=cfg.meta_temperature,
            )
            logger.info(f"Using LabelCorrect (simple 1x1 conv on labels)")
        else:
            # FeatureMapCorrect: separate backbone with image + labels
            correct_backbone = getattr(cfg, 'correct_backbone', cfg.backbone)
            correct_model = FeatureMapCorrect(
                in_channels=cfg.in_channels,
                num_classes=cfg.n_classes + 1,  # +1 for background
                backbone=correct_backbone,
                pretrained=cfg.pretrained,
                decoder_channels=(256, 128, 64, 32, 16),
                spatial_dims=3,
                temperature=cfg.meta_temperature,
            )
            logger.info(f"Using FeatureMapCorrect with backbone={correct_backbone}")

        correct_model.to(device)
        logger.info(f"Correct module parameters: {sum(p.numel() for p in correct_model.parameters())}")

        correct_optimizer = optim.Adam(
            correct_model.parameters(),
            lr=cfg.meta_lr,
            weight_decay=cfg.meta_weight_decay,
        )
        correct_scheduler = build_scheduler(cfg, correct_optimizer, cfg.train_iters)

        correct = Correct(
            name="correct",
            module=correct_model,
            optimizer=correct_optimizer,
            scheduler=correct_scheduler,
            train_data_loader=meta_loader,
            config=meta_config,
            device=device,
            external_config=cfg,
        )

        problems.append(correct)
        l2u[maintask].append(correct)
        u2l[correct] = [maintask]
        logger.info("Correct module created")

    # Setup LossWeight problem if enabled
    loss_weight = None
    if cfg.loss_weight:
        loss_weight_hidden = getattr(cfg, 'loss_weight_hidden_channels', 16)
        loss_weight_layers = getattr(cfg, 'loss_weight_num_layers', 3)

        loss_weight_model = LossWeightModule(
            in_channels=1,  # Total loss per voxel
            hidden_channels=loss_weight_hidden,
            num_layers=loss_weight_layers,
        )
        loss_weight_model.to(device)
        logger.info(f"LossWeight module parameters: {sum(p.numel() for p in loss_weight_model.parameters())}")

        loss_weight_optimizer = optim.Adam(
            loss_weight_model.parameters(),
            lr=cfg.meta_lr,
            weight_decay=cfg.meta_weight_decay,
        )
        loss_weight_scheduler = build_scheduler(cfg, loss_weight_optimizer, cfg.train_iters)

        loss_weight = LossWeight(
            name="loss_weight",
            module=loss_weight_model,
            optimizer=loss_weight_optimizer,
            scheduler=loss_weight_scheduler,
            train_data_loader=meta_loader,
            config=meta_config,
            device=device,
            external_config=cfg,
        )

        problems.append(loss_weight)
        l2u[maintask].append(loss_weight)
        u2l[loss_weight] = [maintask]
        logger.info("LossWeight module created")

    # Setup MetaMixup problem if enabled
    meta_mixup = None
    if cfg.meta_mixup:
        meta_mixup_hidden = getattr(cfg, 'meta_mixup_hidden_channels', 16)
        meta_mixup_layers = getattr(cfg, 'meta_mixup_num_layers', 3)
        meta_mixup_use_targets = getattr(cfg, 'meta_mixup_use_targets', True)

        meta_mixup_model = MetaMixupModule(
            in_channels=cfg.in_channels,
            num_classes=cfg.n_classes + 1,  # +1 for background
            hidden_channels=meta_mixup_hidden,
            num_layers=meta_mixup_layers,
            use_targets=meta_mixup_use_targets,
        )
        meta_mixup_model.to(device)
        logger.info(f"MetaMixup module parameters: {sum(p.numel() for p in meta_mixup_model.parameters())}")

        meta_mixup_optimizer = optim.Adam(
            meta_mixup_model.parameters(),
            lr=cfg.meta_lr,
            weight_decay=cfg.meta_weight_decay,
        )
        meta_mixup_scheduler = build_scheduler(cfg, meta_mixup_optimizer, cfg.train_iters)

        meta_mixup = MetaMixup(
            name="meta_mixup",
            module=meta_mixup_model,
            optimizer=meta_mixup_optimizer,
            scheduler=meta_mixup_scheduler,
            train_data_loader=meta_loader,
            config=meta_config,
            device=device,
            external_config=cfg,
        )

        problems.append(meta_mixup)
        l2u[maintask].append(meta_mixup)
        u2l[meta_mixup] = [maintask]
        logger.info("MetaMixup module created")

    # Setup dependencies
    dependencies = {"l2u": l2u, "u2l": u2l}

    # Create engine
    engine = MetaEngine(
        config=engine_config,
        problems=problems,
        dependencies=dependencies,
        train_loader=train_loader,
        val_loader=val_loader,
        external_config=cfg,
        output_dir=output_dir,
    )

    # Load checkpoint if resuming
    if getattr(cfg, 'resume_checkpoint', None) is not None:
        logger.info(f"Resuming from checkpoint: {cfg.resume_checkpoint}")
        engine.load_checkpoint(cfg.resume_checkpoint)

    # Run training
    logger.info("Starting training...")

    final_metrics = engine.run()

    logger.info(f"Training completed. Final metrics: {final_metrics}")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Meta-learning training for cryo-ET")

    parser.add_argument("-C", "--config", required=True, help="Config file name or path (with or without .py extension)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")

    parser_args, other_args = parser.parse_known_args(sys.argv)

    # Import config
    cfg, config_path = load_config(parser_args.config)

    # Set meta-learning flags
    if cfg.meta_mixup:
        cfg.mixup_p = 0.0  # Disable standard mixup if using meta mixup
        print("Meta mixup enabled, setting cfg.mixup_p = 0.0")
    if parser_args.resume:
        cfg.resume_checkpoint = parser_args.resume

    # Overwrite params in config with additional args
    if len(other_args) > 1:
        other_args = {k.replace('-', ''): v for k, v in zip(other_args[1::2], other_args[2::2])}

        for key in other_args:
            if key in cfg.__dict__:
                print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
                cfg_type = type(cfg.__dict__[key])
                if other_args[key] == 'None':
                    cfg.__dict__[key] = None
                elif cfg_type == bool:
                    cfg.__dict__[key] = other_args[key] == 'True'
                elif cfg_type == type(None):
                    cfg.__dict__[key] = other_args[key]
                else:
                    cfg.__dict__[key] = cfg_type(other_args[key])

    # Import dataset utilities
    cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
    cfg.tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
    cfg.val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
    cfg.batch_to_device = importlib.import_module(cfg.dataset).batch_to_device

    # Import post-processing and metric functions
    cfg.post_process_pipeline = importlib.import_module(cfg.post_process_pipeline).post_process_pipeline
    cfg.calc_metric = importlib.import_module(cfg.metric).calc_metric

    # Run training
    result = train_meta(cfg, parser_args, config_path)
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()
