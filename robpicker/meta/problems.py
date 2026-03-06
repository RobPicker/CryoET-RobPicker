"""
Betty problems for meta-learning based cryo-ET particle picking.

This module defines the bi-level optimization problems:
- MainTask: Lower-level problem (main segmentation model)
- Reweight: Upper-level problem (class weight optimization)
- Correct: Upper-level problem (label correction)
- LossWeight: Upper-level problem (per-voxel loss weight optimization)
- MetaMixup: Upper-level problem (voxel-wise mixup coefficient optimization)
"""

import time
import torch
import torch.nn.functional as F
from betty.problems import ImplicitProblem


def to_ce_target(y):
    """Convert target to cross-entropy format with background channel."""
    y_bg = 1 - y.sum(1, keepdim=True).clamp(0, 1)
    y = torch.cat([y, y_bg], 1)
    # Add epsilon for numerical stability
    y_sum = y.sum(1, keepdim=True).clamp(min=1e-8)
    y = y / y_sum
    return y


def hinge_ce_loss(logits, target, threshold=0.5, class_weights=None, loss_weight_module=None):
    """
    Compute hinge-like cross-entropy loss.

    For each voxel:
    - Get the ground truth class c (argmax of target)
    - If predicted probability for class c > threshold, loss = 0 (already correct)
    - Otherwise, use original cross-entropy loss

    Args:
        logits: Predicted logits [B, C, D, H, W]
        target: Target in CE format (soft labels) [B, C, D, H, W]
        threshold: Probability threshold above which loss is zero
        class_weights: Optional class weights [C] to weight per-class losses

    Returns:
        Scalar loss value
    """
    # Get predicted probabilities
    probs = F.softmax(logits, dim=1)

    # Get ground truth class indices
    gt_class = target.argmax(dim=1)  # [B, D, H, W]

    # Get predicted probability for the ground truth class at each voxel
    # Use gather to select the probability of the gt class
    B, C, D, H, W = probs.shape
    gt_class_expanded = gt_class.unsqueeze(1)  # [B, 1, D, H, W]
    pred_prob_gt = probs.gather(1, gt_class_expanded).squeeze(1)  # [B, D, H, W]

    # Compute per-voxel, per-class cross-entropy loss
    log_probs = F.log_softmax(logits, dim=1)
    per_voxel_per_class_ce = -(target * log_probs)  # [B, C, D, H, W]

    mean_voxel_weight = None
    if loss_weight_module is not None:
        # Sum across classes to get total per-voxel loss [B, 1, D, H, W]
        total_per_voxel_loss = per_voxel_per_class_ce.sum(dim=1, keepdim=True)
        # Get per-voxel weights from the module [B, 1, D, H, W]
        voxel_weights = loss_weight_module(total_per_voxel_loss)
        # Track mean weight for logging
        mean_voxel_weight = voxel_weights.mean()
        # Apply weights to per-voxel loss (broadcast across class dimension)
        per_voxel_per_class_ce = per_voxel_per_class_ce * voxel_weights

    # Apply class weights if provided
    if class_weights is not None:
        # Reshape weights to broadcast: [C] -> [1, C, 1, 1, 1]
        weights = class_weights.view(1, -1, 1, 1, 1)
        per_voxel_per_class_ce = per_voxel_per_class_ce * weights

    # Sum over classes to get per-voxel CE loss
    per_voxel_ce = per_voxel_per_class_ce.sum(dim=1)  # [B, D, H, W]

    # Create mask: 1 where we should apply loss (pred_prob < threshold), 0 otherwise
    loss_mask = (pred_prob_gt < threshold).float()

    # Apply mask to loss
    masked_loss = per_voxel_ce * loss_mask

    # Compute mean loss (only over voxels where mask is 1)
    num_loss_voxels = loss_mask.sum().clamp(min=1.0)
    loss = masked_loss.sum() / num_loss_voxels

    return loss, None, mean_voxel_weight


class BaseProblem(ImplicitProblem):
    """Base problem class with common utilities."""

    def __init__(
        self,
        name,
        config,
        module=None,
        optimizer=None,
        scheduler=None,
        train_data_loader=None,
        device=None,
        external_config=None,
    ):
        super().__init__(
            name, config, module, optimizer, scheduler, train_data_loader, device
        )
        self.external_config = external_config
        self._start_time = time.time()
        self._last_log_step = -1

    def batch_to_device(self, batch):
        """Move batch to device."""
        return {key: batch[key].to(self.device) for key in batch}

    def log_lr(self):
        """Log current learning rate."""
        if self.optimizer is None:
            return
        if self._count % 100 != 0:
            return
        lr = self.optimizer.param_groups[0].get("lr", None)
        if lr is not None:
            self.log({f"{self.name}_lr": lr}, global_step=None)

    def format_time_str(self, seconds):
        days = int(seconds // (24 * 3600))
        hours = int((seconds % (24 * 3600)) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if days > 0:
            return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def log_time(self):
        """Log elapsed time and ETA based on current step."""
        log_step = int(getattr(self.external_config, "log_step", 100))
        if log_step <= 0 or self._count % log_step != 0 or self._count == self._last_log_step:
            return
        self._last_log_step = self._count

        elapsed = time.time() - self._start_time
        total_steps = getattr(self.external_config, "total_steps", None)
        if total_steps is None:
            train_iters = int(getattr(self.external_config, "train_iters", 0))
            unroll_steps = int(getattr(self.external_config, "unroll_steps", 0))
            warmup_steps = int(getattr(self.external_config, "warmup_steps", 0))
            if train_iters > 0 and unroll_steps > 0:
                total_steps = train_iters * unroll_steps + warmup_steps

        if not total_steps:
            return

        current_step = max(int(self._count), 1)
        remaining = max(total_steps - current_step, 0)
        eta = elapsed * (remaining / float(current_step))
        elapsed_str = self.format_time_str(elapsed)
        eta_str = self.format_time_str(eta)
        self.log(
            {
                "elapsed_sec": float(elapsed),
                "eta_sec": float(eta),
            },
            global_step=None,
        )
        print(f'[Problem "{self.name}"] elapsed={elapsed_str} eta={eta_str}')


class MainTask(BaseProblem):
    """
    Main segmentation task (lower-level problem).

    Trains the segmentation model using:
    - Class weights from the reweight module (if enabled)
    - Corrected labels from the correct module (if enabled)
    - Per-voxel loss weights from the loss_weight module (if enabled)
    - Meta-learned mixup coefficients (if enabled)
    """

    def training_step(self, batch):
        batch = self.batch_to_device(batch)
        x = batch['input']
        y = batch['target']

        # Apply meta mixup if enabled
        if hasattr(self, 'meta_mixup') and self.external_config.meta_mixup:
            # Shuffle within batch to get pairs for mixing
            bs = x.shape[0]
            perm = torch.randperm(bs, device=x.device)
            x2 = x[perm]
            y2 = y[perm]

            # Get voxel-wise mixup coefficients from meta module
            coeff = self.meta_mixup(x, x2, to_ce_target(y), to_ce_target(y2))

            # Mix the samples
            x = coeff * x + (1 - coeff) * x2
            y = coeff * y + (1 - coeff) * y2

            # Update batch with mixed data
            batch = {'input': x, 'target': y}

        # Get class weights from reweight module if available
        class_weights = None
        if hasattr(self, 'reweight') and self.external_config.reweight:
            class_weights = self.reweight()

        # Get corrected labels from correct module if available
        corrected_targets = None
        if hasattr(self, 'correct') and self.external_config.correct:
            # Check if module is LabelCorrect (takes only labels) or FeatureMapCorrect (takes image + labels)
            from robpicker.meta.meta_modules import LabelCorrect
            if isinstance(self.correct.module, LabelCorrect):
                # LabelCorrect: only takes labels as input
                _, corrected_targets = self.correct(to_ce_target(y))
            else:
                # FeatureMapCorrect: takes image and labels
                _, corrected_targets = self.correct(x, to_ce_target(y))

        # Get loss weight module if available
        loss_weight_module = None
        if hasattr(self, 'loss_weight') and self.external_config.loss_weight:
            loss_weight_module = self.loss_weight.module

        # Compute loss
        outputs = self.module(
            batch,
            return_features=False,
            class_weights=class_weights,
            corrected_targets=corrected_targets,
            loss_weight_module=loss_weight_module,
        )

        # Log mean voxel weight if available (every 100 steps)
        if 'mean_voxel_weight' in outputs and self._count % 100 == 0:
            self.log({'mean_voxel_weight': outputs['mean_voxel_weight'].item()}, global_step=None)

        self.log_time()
        self.log_lr()
        return outputs['loss']


class Reweight(BaseProblem):
    """
    Class weight reweighting problem (upper-level problem).

    Optimizes class weights to minimize validation loss of the main model.

    Note: The meta dataset (validation split) is assumed to have cleaner labels,
    so we use fixed class weights (from external_config.meta_class_weights)
    when computing the loss on this dataset.
    """

    def training_step(self, batch):
        batch = self.batch_to_device(batch)

        # Use fixed class weights from config for meta dataset (cleaner labels)
        # The meta_class_weights should be set by user in config
        meta_class_weights = None
        if hasattr(self.external_config, 'meta_class_weights') and self.external_config.meta_class_weights is not None:
            meta_class_weights = torch.tensor(
                self.external_config.meta_class_weights,
                dtype=torch.float32,
                device=self.device
            )

        outputs = self.main(
            batch,
            return_features=False,
            class_weights=meta_class_weights,
            loss_weight_module=None,  # Don't use loss weighting for meta objective
        )

        main_loss = outputs['loss']

        self.log_lr()
        return main_loss


class Correct(BaseProblem):
    """
    Label correction problem (upper-level problem).

    Optimizes label correction to minimize validation loss of the main model.

    Uses hinge-like CE loss: if predicted probability for gt class > threshold,
    loss is 0 (already correct), otherwise use original CE loss.

    Supports both:
    - LabelCorrect: Simple 1x1 conv on labels only
    - FeatureMapCorrect: Separate backbone that takes image + labels
    """

    def training_step(self, batch):
        batch = self.batch_to_device(batch)
        x = batch['input']
        y = batch['target']  # Point targets

        # Use fixed class weights from config for meta dataset (cleaner labels)
        meta_class_weights = None
        if hasattr(self.external_config, 'meta_class_weights') and self.external_config.meta_class_weights is not None:
            meta_class_weights = torch.tensor(
                self.external_config.meta_class_weights,
                dtype=torch.float32,
                device=self.device
            )

        outputs = self.main(
            batch,
            return_features=False,
            class_weights=meta_class_weights,
            loss_weight_module=None,  # Don't use loss weighting for meta objective
        )

        main_loss = outputs['loss']

        # Auxiliary loss: train corrector
        if self.external_config.meta_lambda > 0:
            # Check if module is LabelCorrect (takes only labels) or FeatureMapCorrect (takes image + labels)
            from robpicker.meta.meta_modules import LabelCorrect
            if isinstance(self.module, LabelCorrect):
                # LabelCorrect: only takes labels as input
                correct_logits, correct_probs = self.module(to_ce_target(y))
            else:
                # FeatureMapCorrect: takes image and labels
                correct_logits, correct_probs = self.module(x, to_ce_target(y))

            # Cross-entropy loss between corrector predictions and targets
            aux_loss = F.cross_entropy(
                correct_logits,
                to_ce_target(y),
                reduction='mean'
            )

            total_loss = main_loss + self.external_config.meta_lambda * aux_loss
        else:
            total_loss = main_loss

        self.log_lr()
        return total_loss


class LossWeight(BaseProblem):
    """
    Per-voxel loss weighting problem (upper-level problem).

    Optimizes a module that produces per-voxel weights for the loss.
    The module takes per-voxel loss as input and outputs weights in [0, 1].
    These weights are multiplied with the loss before aggregation.

    The training step evaluates the main model on the meta dataset (with cleaner labels)
    using the learned voxel weights, similar to how Reweight evaluates with class weights.
    """

    def training_step(self, batch):
        batch = self.batch_to_device(batch)

        # Use fixed class weights from config for meta dataset (cleaner labels)
        meta_class_weights = None
        if hasattr(self.external_config, 'meta_class_weights') and self.external_config.meta_class_weights is not None:
            meta_class_weights = torch.tensor(
                self.external_config.meta_class_weights,
                dtype=torch.float32,
                device=self.device
            )

        outputs = self.main(
            batch,
            return_features=False,
            class_weights=meta_class_weights,
            loss_weight_module=None,  # Don't use loss weighting for meta objective
        )

        main_loss = outputs['loss']

        self.log_lr()
        return main_loss


class MetaMixup(BaseProblem):
    """
    Meta-learned voxel-wise mixup coefficient problem (upper-level problem).

    Optimizes a module that produces voxel-wise mixup coefficients.
    The training step evaluates the main model on the meta dataset WITHOUT mixup,
    using standard loss computation (no mixing applied).

    The meta mixup module is used in the lower-level (MainTask) to mix training
    samples with learned coefficients. The upper-level objective is to minimize
    validation loss without mixup, so the meta module learns coefficients that
    improve generalization.
    """

    def training_step(self, batch):
        batch = self.batch_to_device(batch)

        # Use fixed class weights from config for meta dataset (cleaner labels)
        meta_class_weights = None
        if hasattr(self.external_config, 'meta_class_weights') and self.external_config.meta_class_weights is not None:
            meta_class_weights = torch.tensor(
                self.external_config.meta_class_weights,
                dtype=torch.float32,
                device=self.device
            )

        # Forward pass without any mixup - standard evaluation on meta dataset
        outputs = self.main(
            batch,
            return_features=False,
            class_weights=meta_class_weights,
            loss_weight_module=None,  # Don't use loss weighting for meta objective
        )

        main_loss = outputs['loss']

        self.log_lr()
        return main_loss
