"""
Config for EMPIAR-style dataset (dataset-specific settings only).
"""

from copy import copy
import os
import numpy as np

from robpicker.configs.meta_config import meta_cfg

cfg = copy(meta_cfg)

# Paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"./output/{os.path.basename(__file__).split('.')[0]}"

# Dataset-specific settings
cfg.data_dir = "/empiar_11830_data"
cfg.train_folder = "train"
cfg.meta_folder = "meta"
cfg.test_folder = "test"
cfg.voxel_spacing = 7.84

# Particle classes
cfg.classes = ["ribosome80s", "atp"]
cfg.n_classes = len(cfg.classes)
cfg.class_mapping = {
    1: "ribosome80s",
    2: "atp",
}

# Particle radii (Angstroms)
cfg.particle_radi = {
    "ribosome80s": 150,
    "atp": 80,
}

# Post-processing bounds for tomograms
cfg.pp_x_max = 10500
cfg.pp_y_max = 10500
cfg.pp_z_max = 5500
cfg.pp_conf_thresh = 0.01

cfg.metric_beta = 1
cfg.metric_distance_multiplier = 0.5
cfg.metric_weights = {
    "ribosome80s": 1,
    "atp": 1,
}

# Model configuration tied to class count
cfg.backbone_args = dict(
    spatial_dims=3,
    in_channels=cfg.in_channels,
    out_channels=cfg.n_classes,
    backbone=cfg.backbone,
    pretrained=cfg.pretrained,
)

cfg.lr = 5e-4
cfg.unroll_steps = 5     # Number of inner loop steps before meta-update
cfg.warmup_steps = 50    # Warmup steps before meta-learning starts
cfg.train_iters = 5000   # Total training iterations per epoch
cfg.valid_steps = 500    # Validation frequency (steps)
cfg.log_step = 20       # Logging frequency

cfg.meta_lr = 2e-4

# Class weights (initial values, will be learned if --reweight is enabled)
cfg.class_weights = np.array([64, 512, 1])
cfg.lvl_weights = np.array([0, 0, 0, 1])
cfg.meta_class_weights = np.array([64, 512, 1])

# Resampling settings for class-aware crops
cfg.resample_weight = [1.0, 2.0]  # adjust this to make the less frequent classes more likely to be sampled
cfg.resample_bg_weight = 0.1
cfg.resample_stats_batches = 30  # use this to check sample frequency at the beginning of training so you can adjust resample weights if needed

default_cfg = cfg
