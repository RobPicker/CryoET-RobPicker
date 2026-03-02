import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.flexible_unet import UNetDecoder, FLEXUNET_BACKBONE


class ClassWeightReweight(nn.Module):
    """
    Simple learnable class-weight vector for data reweighting.

    Uses a learnable parameter vector that is passed through softplus
    to ensure positive weights.
    """
    def __init__(self, num_classes=7, init_weights=None):
        super(ClassWeightReweight, self).__init__()

        self.num_classes = num_classes

        # Initialize with provided weights or uniform
        if init_weights is not None:
            init_weights = torch.from_numpy(init_weights).float()
        else:
            init_weights = torch.ones(num_classes)

        self.weight_logits = nn.Parameter(init_weights)

    def forward(self):
        """
        Returns the class weights (positive values via softplus).
        """
        return self.weight_logits

    def get_normalized_weights(self):
        """
        Returns normalized class weights that sum to num_classes.
        """
        weights = self.forward()
        return weights * self.num_classes / weights.sum().clamp(min=1e-8)


class LabelCorrect(nn.Module):
    """
    Simple label correction module that directly transforms labels.

    Takes the original label (after to_ce_target) as input and outputs
    corrected labels using a simple 1x1 convolution.
    """
    def __init__(self, num_classes: int = 7, hidden_channels: int = 16, temperature: float = 1.0):
        """
        Args:
            num_classes: Number of classes (including background)
            hidden_channels: Number of hidden channels in the conv layers
            temperature: Temperature for softmax
        """
        super(LabelCorrect, self).__init__()

        self.num_classes = num_classes
        self.temperature = temperature

        self.conv = nn.Sequential(
            nn.Conv3d(num_classes, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, num_classes, kernel_size=3, padding=1),
        )
        self.alpha = nn.Parameter(-2.0 * torch.ones(1))

        print(f"LabelCorrect: num_classes={num_classes}, hidden_channels={hidden_channels}")

    def forward(self, labels):
        """
        Args:
            labels: Ground truth labels in CE format [B, num_classes, D, H, W]

        Returns:
            logits: Corrected logits [B, num_classes, D, H, W]
            probs: Corrected probabilities [B, num_classes, D, H, W]
        """
        logits = self.conv(labels)
        pred = torch.softmax(logits / self.temperature, dim=1)
        # Use sigmoid to constrain alpha to [0, 1] range for stable gradients
        alpha = torch.sigmoid(self.alpha)
        probs = alpha * pred + (1 - alpha) * labels

        return logits, probs


class LossWeightModule(nn.Module):
    """
    Per-voxel loss weighting module (MLP-style).

    Takes the per-voxel loss map as input and outputs a weight (0-1) for each voxel.
    The weights are multiplied with the loss before aggregation.

    Architecture: Per-voxel MLP using 1x1 Conv3d layers. Each voxel's weight
    depends only on that voxel's loss value, not on neighboring voxels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        num_layers: int = 3,
    ):
        """
        Args:
            in_channels: Number of input channels (typically 1 for loss map,
                         or num_classes for per-class loss)
            hidden_channels: Number of hidden channels in MLP
            num_layers: Number of layers (minimum 2: input -> hidden -> output)
        """
        super(LossWeightModule, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        num_layers = max(2, num_layers)  # At least 2 layers

        # Build per-voxel MLP using 1x1 convolutions
        layers = []

        # First layer: in_channels -> hidden_channels
        layers.append(nn.Conv3d(in_channels, hidden_channels, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            layers.append(nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: hidden_channels -> 1
        layers.append(nn.Conv3d(hidden_channels, 1, kernel_size=1))
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1]

        self.network = nn.Sequential(*layers)

        # Initialize to output ~1.0 (uniform weighting initially)
        # By setting the last conv bias to a positive value
        with torch.no_grad():
            # Find the last conv layer and initialize its bias
            for module in reversed(list(self.network.modules())):
                if isinstance(module, nn.Conv3d):
                    # sigmoid(2) ≈ 0.88, start with high weights
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 2.0)
                    break

        print(f"LossWeightModule (MLP): in_channels={in_channels}, hidden_channels={hidden_channels}, num_layers={num_layers}")

    def forward(self, loss_map):
        """
        Args:
            loss_map: Per-voxel loss tensor [B, C, D, H, W] or [B, 1, D, H, W]

        Returns:
            weights: Per-voxel weights in [0, 1], shape [B, 1, D, H, W]
        """
        # If loss_map has multiple channels (per-class loss), sum to get total loss per voxel
        if loss_map.shape[1] > 1:
            loss_map = loss_map.sum(dim=1, keepdim=True)

        weights = self.network(loss_map)
        return weights


class FeatureMapCorrect(nn.Module):
    """
    Label correction module with its own separate backbone.

    Has its own U-Net encoder-decoder to extract features from the input image,
    then combines with ground truth labels to output corrected labels.
    This does NOT share weights with the main segmentation network.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        backbone: str = "resnet34",
        pretrained: bool = False,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        spatial_dims: int = 3,
        temperature: float = 1.0,
    ):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale)
            num_classes: Number of output classes (including background)
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained encoder weights
            decoder_channels: Decoder channel configuration
            spatial_dims: Spatial dimensions (2 or 3)
            temperature: Temperature for softmax
        """
        super(FeatureMapCorrect, self).__init__()

        self.num_classes = num_classes
        self.temperature = temperature
        self.spatial_dims = spatial_dims

        # Build encoder
        if backbone not in FLEXUNET_BACKBONE.register_dict:
            raise ValueError(
                f"invalid backbone {backbone}, must be one of {FLEXUNET_BACKBONE.register_dict.keys()}."
            )

        encoder_info = FLEXUNET_BACKBONE.register_dict[backbone]
        encoder_parameters = encoder_info["parameter"]
        encoder_parameters.update({
            "spatial_dims": spatial_dims,
            "in_channels": in_channels,
            "pretrained": pretrained
        })
        encoder_type = encoder_info["type"]
        self.encoder = encoder_type(**encoder_parameters)

        encoder_feature_num = encoder_info["feature_number"]
        encoder_channels = tuple([in_channels] + list(encoder_info["feature_channel"]))
        decoder_channels = decoder_channels[:encoder_feature_num]
        self.skip_connect = encoder_feature_num - 1

        # Build decoder
        self.decoder = UNetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            act=("relu", {"inplace": True}),
            norm=("batch", {"eps": 1e-3, "momentum": 0.1}),
            dropout=0.0,
            bias=False,
            upsample="nontrainable",
            interp_mode="nearest",
            pre_conv="default",
            align_corners=None,
            is_pad=True,
        )

        # Feature channels from decoder output (last decoder channel)
        feature_channels = decoder_channels[-1]

        # Classifier head that combines features with labels
        self.classifier = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier2 = nn.Sequential(
            nn.Conv3d(feature_channels + num_classes, feature_channels, kernel_size=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, num_classes, kernel_size=1)
        )

        print(f"FeatureMapCorrect: backbone={backbone}, encoder_channels={encoder_channels}, "
              f"decoder_channels={decoder_channels}, feature_channels={feature_channels}")

    def forward(self, x, labels):
        """
        Args:
            x: Input image [B, C, D, H, W]
            labels: Ground truth labels [B, num_classes, D, H, W]

        Returns:
            logits: Predicted logits [B, num_classes, D, H, W]
            probs: Predicted probabilities (corrected labels) [B, num_classes, D, H, W]
        """
        # Extract features using own backbone
        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out, self.skip_connect)
        features = decoder_out  # Final decoder output

        # Process features
        feat = self.classifier(features)

        # Resize labels to match feature map spatial dimensions if needed
        if labels.shape[2:] != feat.shape[2:]:
            labels = F.interpolate(labels, size=feat.shape[2:], mode='trilinear', align_corners=False)

        # Combine features with labels
        feat_and_labels = torch.cat([feat, labels], dim=1)
        logits = self.classifier2(feat_and_labels)
        probs = torch.softmax(logits / self.temperature, dim=1)

        return logits, probs


class MetaMixupModule(nn.Module):
    """
    Meta-learned voxel-wise mixup coefficient module.

    Takes two samples (images) and their targets as input, and outputs a
    voxel-wise coefficient map in [0, 1] for mixing the samples.

    mixed_x = coeff * x1 + (1 - coeff) * x2
    mixed_y = coeff * y1 + (1 - coeff) * y2

    Architecture: A lightweight CNN that processes the concatenation of two
    images and optionally their targets to produce a spatial coefficient map.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        hidden_channels: int = 16,
        num_layers: int = 3,
        use_targets: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels per image (typically 1)
            num_classes: Number of classes in targets (excluding background)
            hidden_channels: Number of hidden channels in the network
            num_layers: Number of conv layers
            use_targets: Whether to also use targets as input (in addition to images)
        """
        super(MetaMixupModule, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.use_targets = use_targets

        # Input: concatenation of two images (and optionally two targets)
        # Images: 2 * in_channels
        # Targets (if used): 2 * num_classes
        if use_targets:
            total_in_channels = 2 * in_channels + 2 * num_classes
        else:
            total_in_channels = 2 * in_channels

        # Build the network with spatial context (use 3x3 convolutions)
        layers = []

        # First layer
        layers.append(nn.Conv3d(total_in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm3d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: hidden_channels -> 1 (coefficient map)
        layers.append(nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1]

        self.network = nn.Sequential(*layers)

        # Initialize to output ~0.5 (uniform mixing initially)
        # with torch.no_grad():
        #     for module in reversed(list(self.network.modules())):
        #         if isinstance(module, nn.Conv3d):
        #             nn.init.zeros_(module.weight)
        #             if module.bias is not None:
        #                 nn.init.constant_(module.bias, 0.0)  # sigmoid(0) = 0.5
        #             break

        print(f"MetaMixupModule: in_channels={in_channels}, num_classes={num_classes}, "
              f"hidden_channels={hidden_channels}, num_layers={num_layers}, use_targets={use_targets}")

    def forward(self, x1, x2, y1=None, y2=None):
        """
        Args:
            x1: First image [B, C, D, H, W]
            x2: Second image [B, C, D, H, W]
            y1: First target [B, num_classes, D, H, W] (optional if use_targets=False)
            y2: Second target [B, num_classes, D, H, W] (optional if use_targets=False)

        Returns:
            coeff: Voxel-wise mixing coefficient [B, 1, D, H, W] in [0, 1]
        """
        if self.use_targets:
            if y1 is None or y2 is None:
                raise ValueError("Targets y1 and y2 are required when use_targets=True")
            # Concatenate images and targets
            inputs = torch.cat([x1, x2, y1, y2], dim=1)
        else:
            # Concatenate only images
            inputs = torch.cat([x1, x2], dim=1)

        coeff = self.network(inputs)
        return coeff

    def mix_samples(self, x1, x2, y1, y2, coeff=None):
        """
        Mix two samples using the learned coefficient map.

        Args:
            x1: First image [B, C, D, H, W]
            x2: Second image [B, C, D, H, W]
            y1: First target [B, num_classes, D, H, W]
            y2: Second target [B, num_classes, D, H, W]
            coeff: Pre-computed coefficient map (optional). If None, computed from inputs.

        Returns:
            mixed_x: Mixed image [B, C, D, H, W]
            mixed_y: Mixed target [B, num_classes, D, H, W]
            coeff: The coefficient map used [B, 1, D, H, W]
        """
        if coeff is None:
            coeff = self.forward(x1, x2, y1, y2)

        # Mix the samples
        mixed_x = coeff * x1 + (1 - coeff) * x2
        mixed_y = coeff * y1 + (1 - coeff) * y2

        return mixed_x, mixed_y, coeff
