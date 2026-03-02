"""
Meta-learning enabled model for cryo-ET particle picking.

This module support:
- Returning penultimate feature maps for label correction
- External class weights for meta-learned reweighting
- Label correction via external corrected targets
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from monai.networks.nets.flexible_unet import SegmentationHead, UNetDecoder, FLEXUNET_BACKBONE


class PatchedUNetDecoder(UNetDecoder):
    """UNet decoder that outputs all intermediate feature maps."""

    def forward(self, features: list[torch.Tensor], skip_connect: int = 4):
        skips = features[:-1][::-1]
        features = features[1:][::-1]

        out = []
        x = features[0]
        out += [x]
        for i, block in enumerate(self.blocks):
            if i < skip_connect:
                skip = skips[i]
            else:
                skip = None
            x = block(x, skip)
            out += [x]
        return out


class FlexibleUNetMeta(nn.Module):
    """
    Flexible UNet implementation with meta-learning support.

    Can optionally return the penultimate decoder feature map for use
    in label correction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str,
        pretrained: bool = False,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        spatial_dims: int = 2,
        norm: str | tuple = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: str | tuple = ("relu", {"inplace": True}),
        dropout: float | tuple = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        pre_conv: str = "default",
        interp_mode: str = "nearest",
        is_pad: bool = True,
    ) -> None:
        super().__init__()

        if backbone not in FLEXUNET_BACKBONE.register_dict:
            raise ValueError(
                f"invalid model_name {backbone} found, must be one of {FLEXUNET_BACKBONE.register_dict.keys()}."
            )

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")

        encoder = FLEXUNET_BACKBONE.register_dict[backbone]
        self.backbone = backbone
        self.spatial_dims = spatial_dims
        encoder_parameters = encoder["parameter"]
        if not (
            ("spatial_dims" in encoder_parameters)
            and ("in_channels" in encoder_parameters)
            and ("pretrained" in encoder_parameters)
        ):
            raise ValueError("The backbone init method must have spatial_dims, in_channels and pretrained parameters.")
        encoder_feature_num = encoder["feature_number"]
        if encoder_feature_num > 5:
            raise ValueError("Flexible unet can only accept no more than 5 encoder feature maps.")

        decoder_channels = decoder_channels[:encoder_feature_num]
        self.skip_connect = encoder_feature_num - 1
        encoder_parameters.update({"spatial_dims": spatial_dims, "in_channels": in_channels, "pretrained": pretrained})
        encoder_channels = tuple([in_channels] + list(encoder["feature_channel"]))
        encoder_type = encoder["type"]
        self.encoder = encoder_type(**encoder_parameters)
        print(f"Decoder channels: {decoder_channels}")

        self.decoder = PatchedUNetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=pre_conv,
            align_corners=None,
            is_pad=is_pad,
        )

        # Store decoder channel info for meta modules
        self.decoder_channels = decoder_channels
        # decoder_out = decoder output[1:-1] -> corresponds to decoder_channels[:-1]
        # decoder_out[-2] is the penultimate feature, which has decoder_channels[:-1][-2] channels
        self.output_decoder_channels = decoder_channels[:-1]  # Channels of decoder_out

        self.segmentation_heads = nn.ModuleList([
            SegmentationHead(
                spatial_dims=spatial_dims,
                in_channels=decoder_channel,
                out_channels=out_channels + 1,
                kernel_size=3,
                act=None,
            ) for decoder_channel in decoder_channels[:-1]
        ])

    def forward(self, inputs: torch.Tensor, return_features: bool = False):
        """
        Forward pass.

        Args:
            inputs: Input tensor [B, C, D, H, W]
            return_features: If True, also return the penultimate decoder feature map

        Returns:
            If return_features=False: List of segmentation outputs at different scales
            If return_features=True: (outputs, penultimate_features)
        """
        x = inputs
        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out, self.skip_connect)[1:-1]
        x_seg = [self.segmentation_heads[i](decoder_out[i]) for i in range(len(decoder_out))]

        if return_features:
            # Return penultimate feature map (second to last in decoder_out)
            penultimate_features = decoder_out[-2]
            return x_seg, penultimate_features

        return x_seg


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


class DenseCrossEntropyMeta(nn.Module):
    """
    Dense cross-entropy loss with external class weights support.

    Can accept class weights from an external source (meta-learned weights).
    Also supports per-voxel loss weighting via loss_weight_module.
    """
    def __init__(self, default_class_weights=None):
        super(DenseCrossEntropyMeta, self).__init__()
        self.default_class_weights = default_class_weights

    def forward(self, x, target, class_weights=None, loss_weight_module=None):
        """
        Args:
            x: Predictions [B, C, D, H, W]
            target: Targets [B, C, D, H, W]
            class_weights: Optional external class weights. If None, uses default.
            loss_weight_module: Optional module that takes per-voxel loss and outputs weights.

        Returns:
            loss: Scalar loss value
            class_losses: Per-class loss values
            mean_voxel_weight: Mean voxel weight (None if loss_weight_module not used)
        """
        x = x.float()
        target = target.float()

        # Clamp target to avoid numerical issues
        target = target.clamp(min=0, max=1)

        logprobs = torch.nn.functional.log_softmax(x, dim=1, dtype=torch.float)

        # Per-voxel, per-class loss [B, C, D, H, W]
        per_voxel_loss = -logprobs * target

        # Apply loss weight module if provided
        mean_voxel_weight = None
        if loss_weight_module is not None:
            # Sum across classes to get total per-voxel loss [B, 1, D, H, W]
            total_per_voxel_loss = per_voxel_loss.sum(dim=1, keepdim=True)
            # Get per-voxel weights from the module [B, 1, D, H, W]
            voxel_weights = loss_weight_module(total_per_voxel_loss.detach().clone())
            # Track mean weight for logging
            mean_voxel_weight = voxel_weights.mean()
            # Apply weights to per-voxel loss (broadcast across class dimension)
            per_voxel_loss = per_voxel_loss * voxel_weights

        # Compute class losses (mean over batch and spatial dims)
        class_losses = per_voxel_loss.mean((0, 2, 3, 4))

        # Use external weights if provided, otherwise use default
        weights = class_weights if class_weights is not None else self.default_class_weights

        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.float().to(class_losses.device)
                loss = (class_losses * weights).sum()
            else:
                loss = (class_losses * torch.tensor(weights, device=class_losses.device, dtype=torch.float32)).sum()
        else:
            loss = class_losses.sum()

        # Check for NaN and replace with 0 (with warning)
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected! class_losses={class_losses}, weights={weights}")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        return loss, class_losses, mean_voxel_weight


class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Z=None):
        bs = X.shape[0]
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        X_coeffs = coeffs.view((-1,) + (1,)*(X.ndim-1))
        Y_coeffs = coeffs.view((-1,) + (1,)*(Y.ndim-1))

        X = X_coeffs * X + (1-X_coeffs) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y[perm]

        if Z:
            return X, Y, Z

        return X, Y


def to_ce_target(y):
    """Convert target to cross-entropy format with background channel."""
    y_bg = 1 - y.sum(1, keepdim=True).clamp(0, 1)
    y = torch.cat([y, y_bg], 1)
    # Add epsilon for numerical stability
    y_sum = y.sum(1, keepdim=True).clamp(min=1e-8)
    y = y / y_sum
    return y


class NetMeta(nn.Module):
    """
    Main network for meta-learning based training.

    Supports:
    - External class weights for loss reweighting
    - Returning penultimate features for label correction
    - Label correction via external corrected targets
    """

    def __init__(self, cfg):
        super(NetMeta, self).__init__()

        self.cfg = cfg
        self.n_classes = cfg.n_classes
        self.classes = cfg.classes

        self.backbone = FlexibleUNetMeta(**cfg.backbone_args)

        # Store penultimate feature channel count for label correction module
        # decoder_out[-2] has output_decoder_channels[-2] channels
        self.penultimate_channels = self.backbone.output_decoder_channels[-2]

        self.mixup = Mixup(cfg.mixup_beta)

        print(f'NetMeta parameters: {human_format(count_parameters(self))}')
        self.lvl_weights = torch.from_numpy(cfg.lvl_weights)

        # Use default class weights if provided, but support external override
        default_weights = None
        if hasattr(cfg, 'class_weights') and cfg.class_weights is not None:
            default_weights = torch.from_numpy(cfg.class_weights)
        self.loss_fn = DenseCrossEntropyMeta(default_class_weights=default_weights)

    def forward(
        self,
        batch,
        return_features: bool = False,
        class_weights: torch.Tensor = None,
        corrected_targets: torch.Tensor = None,
        logits_only: bool = False,
        loss_weight_module: nn.Module = None,
    ):
        """
        Forward pass with meta-learning support.

        Args:
            batch: Input batch dictionary
            return_features: If True, return penultimate features
            class_weights: External class weights (from meta reweight module)
            corrected_targets: Corrected labels (from meta correct module)
            logits_only: If True, only return logits without computing loss
            loss_weight_module: Optional module for per-voxel loss weighting

        Returns:
            outputs dict, optionally with 'penultimate_features'
        """
        x = batch['input']
        y = batch.get("target", None)

        # Apply mixup during training
        if self.training and (y is not None or corrected_targets is not None):
            if torch.rand(1)[0] < self.cfg.mixup_p:
                if self.cfg.meta_mixup:
                    raise RuntimeError("Mixup should be disabled when using meta mixup.")
                if corrected_targets is not None:
                    x, corrected_targets = self.mixup(x, corrected_targets)
                else:
                    x, y = self.mixup(x, y)

        # Forward through backbone
        if return_features:
            out, penultimate_features = self.backbone(x, return_features=True)
        else:
            out = self.backbone(x, return_features=False)
            penultimate_features = None

        outputs = {}

        # Always return logits for hinge loss computation
        outputs["logits"] = out[-1]
        if logits_only:
            return outputs

        # Use corrected targets if provided, otherwise use original
        mean_voxel_weight = None
        if corrected_targets is not None:
            # Compute loss at each scale
            ys = [F.adaptive_max_pool3d(corrected_targets, item.shape[-3:]) for item in out]
            loss_results = [
                self.loss_fn(out[i], ys[i], class_weights=class_weights, loss_weight_module=loss_weight_module)
                for i in range(len(out))
            ]
            losses = torch.stack([r[0] for r in loss_results])
            # Get mean_voxel_weight from last scale (typically the one with weight=1)
            mean_voxel_weight = loss_results[-1][2]
        elif y is not None:
            # Compute loss at each scale
            ys = [F.adaptive_max_pool3d(y, item.shape[-3:]) for item in out]
            loss_results = [
                self.loss_fn(out[i], to_ce_target(ys[i]), class_weights=class_weights, loss_weight_module=loss_weight_module)
                for i in range(len(out))
            ]
            losses = torch.stack([r[0] for r in loss_results])
            # Get mean_voxel_weight from last scale (typically the one with weight=1)
            mean_voxel_weight = loss_results[-1][2]

        if corrected_targets is not None or y is not None:
            lvl_weights = self.lvl_weights.to(losses.device)
            loss = (losses * lvl_weights).sum() / lvl_weights.sum()
            outputs['loss'] = loss
            if mean_voxel_weight is not None:
                outputs['mean_voxel_weight'] = mean_voxel_weight

        if not self.training:
            if 'location' in batch:
                outputs["location"] = batch['location']

        if return_features:
            outputs['penultimate_features'] = penultimate_features

        return outputs


# For backward compatibility with existing training scripts
Net = NetMeta
