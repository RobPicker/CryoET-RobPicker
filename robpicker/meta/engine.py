"""
Betty engine for meta-learning based cryo-ET particle picking.

This module provides a custom engine that extends Betty's base Engine
with validation and checkpointing capabilities.
"""

import os
import math
import logging
import torch
import numpy as np
from betty.engine import Engine
from collections import defaultdict
from tqdm import tqdm


def to_ce_target(y):
    """Convert target to cross-entropy format with background channel."""
    y_bg = 1 - y.sum(1, keepdim=True).clamp(0, 1)
    y = torch.cat([y, y_bg], 1)
    y = y / y.sum(1, keepdim=True)
    return y


class MetaEngine(Engine):
    """
    Custom Betty engine with validation and checkpointing.

    Extends Betty's Engine to add:
    - Custom validation on validation set with checkpointing
    - Best checkpoint saving based on validation score (F-beta)
    - Logging of training progress
    - Full metric calculation using post-processing pipeline

    Note: Betty already handles periodic validation via valid_step config.
    We override validation() to customize the validation behavior.
    """

    def __init__(
        self,
        problems,
        config=None,
        dependencies=None,
        env=None,
        train_loader=None,
        val_loader=None,
        external_config=None,
        output_dir=None,
    ):
        super().__init__(
            problems=problems,
            config=config,
            dependencies=dependencies,
            env=env,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.external_config = external_config
        self.output_dir = output_dir

        # Get device from main model
        self.device = self.main.trainable_parameters()[0].device

        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_score = -float('inf')  # For F-beta score (higher is better)
        self.current_epoch = 0
        self.global_step = 0

        # Setup logging
        self._logger = logging.getLogger(__name__)

    def extract_checkpoint(self):
        """Extract checkpoint from all problems."""
        checkpoint = {}
        checkpoint["main"] = self.main.module.state_dict()
        if hasattr(self, "reweight"):
            checkpoint["reweight"] = self.reweight.module.state_dict()
        if hasattr(self, "correct"):
            checkpoint["correct"] = self.correct.module.state_dict()
        if hasattr(self, "loss_weight"):
            checkpoint["loss_weight"] = self.loss_weight.module.state_dict()
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["best_val_score"] = self.best_val_score
        checkpoint["current_epoch"] = self.current_epoch
        return checkpoint

    def save_checkpoint(self, path=None, name="checkpoint.pth"):
        """Save checkpoint."""
        checkpoint = self.extract_checkpoint()
        if path is None:
            path = self.output_dir
        if path is not None:
            os.makedirs(path, exist_ok=True)
            torch.save(checkpoint, os.path.join(path, name))
            self._logger.info(f"Saved checkpoint to {os.path.join(path, name)}")

    def save_best_checkpoint(self, path=None):
        """Save best checkpoint."""
        self.save_checkpoint(path, "checkpoint_best.pth")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint from file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.main.module.load_state_dict(checkpoint["main"])
        if hasattr(self, "reweight") and "reweight" in checkpoint:
            self.reweight.module.load_state_dict(checkpoint["reweight"])
        if hasattr(self, "correct") and "correct" in checkpoint:
            self.correct.module.load_state_dict(checkpoint["correct"])
        if hasattr(self, "loss_weight") and "loss_weight" in checkpoint:
            self.loss_weight.module.load_state_dict(checkpoint["loss_weight"])
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]
        if "best_val_score" in checkpoint:
            self.best_val_score = checkpoint["best_val_score"]
        if "current_epoch" in checkpoint:
            self.current_epoch = checkpoint["current_epoch"]
        self._logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def do_validation(self):
        """Override to always do validation when called."""
        return self.val_loader is not None

    @torch.no_grad()
    def validation(self):
        """
        Run validation and return metrics.

        Adapted from run_eval in train.py to calculate F-beta scores.
        This method is called by Betty's Engine.run() based on valid_step config.
        Returns a dict that Betty will log.
        """
        if self.val_loader is None:
            return {}

        cfg = self.external_config
        self.main.module.eval()

        # Store information for evaluation
        val_data = defaultdict(list)
        # import ipdb; ipdb.set_trace()

        for batch in tqdm(self.val_loader, desc="Validation", disable=False):
            batch = {key: batch[key].to(self.device) for key in batch}

            # Get class weights if using reweight module
            class_weights = None
            if hasattr(self, 'reweight') and cfg.reweight:
                class_weights = self.reweight.module()

            # Forward pass
            outputs = self.main.module(
                batch,
                return_features=False,
                class_weights=class_weights
            )

            # Collect outputs
            for key, val in outputs.items():
                if isinstance(val, torch.Tensor):
                    val_data[key].append(val.detach().cpu())
                elif isinstance(val, list):
                    val_data[key].extend(val)
                else:
                    val_data[key].append(val)
            if 'experiment_idx' in batch:
                val_data['experiment_idx'].append(batch['experiment_idx'].detach().cpu())
            # val_data['target'].append(to_ce_target(batch['target']).detach().cpu())
            # val_data['input'].append(to_ce_target(batch['input']).detach().cpu())

        # Concatenate/stack collected outputs
        for key, value in val_data.items():
            if isinstance(value[0], torch.Tensor):
                if len(value[0].shape) == 0:
                    val_data[key] = torch.stack(value)
                else:
                    val_data[key] = torch.cat(value, dim=0)
        # val_data['logits'] = val_data['target'] * 10
        # Compute mean validation loss
        mean_val_loss = float('inf')
        if 'loss' in val_data:
            losses = val_data['loss'].cpu().numpy()
            mean_val_loss = np.mean(losses)
            if not math.isnan(mean_val_loss):
                self._logger.info(f"Mean val_loss: {mean_val_loss:.4f}")

        # Calculate metrics if calc_metric is enabled and required functions are available
        val_score = {}
        if callable(getattr(cfg, 'calc_metric', None)) and callable(getattr(cfg, 'post_process_pipeline', None)):
            # try:
            if True:
                val_df = self.val_loader.dataset.df

                # Pass experiment names to cfg for multi-experiment support in post-processing
                if hasattr(self.val_loader.dataset, 'val_experiment_names'):
                    cfg.val_experiment_names = self.val_loader.dataset.val_experiment_names

                pp_out = cfg.post_process_pipeline(cfg, val_data, val_df)
                val_score = cfg.calc_metric(cfg, pp_out, val_df, "val")

                if not isinstance(val_score, dict):
                    val_score = {'score': val_score}

                for k, v in val_score.items():
                    if not math.isnan(v) and not math.isinf(v):
                        self._logger.info(f"val_{k}: {v:.4f}")

            # except Exception as e:
            #     self._logger.warning(f"Failed to calculate metrics: {e}")

        # Determine if this is the best model (by score if available, otherwise by loss)
        main_score = val_score.get('score', None)
        is_best = False

        if main_score is not None and not math.isnan(main_score):
            # Higher score is better
            if main_score > self.best_val_score:
                self.best_val_score = main_score
                is_best = True
                self._logger.info(f"New best score: {self.best_val_score:.4f}")
        else:
            # Lower loss is better
            if mean_val_loss < self.best_val_loss:
                self.best_val_loss = mean_val_loss
                is_best = True
                self._logger.info(f"New best loss: {self.best_val_loss:.4f}")

        if is_best:
            self.save_best_checkpoint(self.output_dir)
        else:
            self.save_checkpoint(self.output_dir, f"checkpoint_step{self.global_step}.pth")

        self.main.module.train()

        # Return metrics for Betty's logging
        result = {
            "val_loss": mean_val_loss,
            "best_val_loss": self.best_val_loss,
        }
        if main_score is not None:
            result["val_score"] = main_score
            result["best_val_score"] = self.best_val_score

        # Add per-particle scores if available
        for k, v in val_score.items():
            if k != 'score':
                result[f"val_{k}"] = v

        return result

    def run(self):
        """Run training with validation callbacks."""
        self._logger.info("Starting meta-learning training...")
        self._logger.info(f"External config: reweight={self.external_config.reweight}, correct={self.external_config.correct}")

        # Run Betty training loop (includes periodic validation via valid_step)
        super().run()

        # Final validation
        final_metrics = self.validation()

        # Save final checkpoint
        self.save_checkpoint(self.output_dir, "checkpoint_last.pth")

        self._logger.info("Training completed!")
        if self.best_val_score > -float('inf'):
            self._logger.info(f"Best validation score: {self.best_val_score:.4f}")
        self._logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return final_metrics
