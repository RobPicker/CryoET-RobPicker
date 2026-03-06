from types import SimpleNamespace
from monai import transforms as mt

cfg = SimpleNamespace(**{})

# stages
cfg.train = True
cfg.val = True
cfg.test = False
cfg.train_val = False

# training routine
cfg.optimizer = "Adam"
cfg.lr = 1e-4
cfg.schedule = "cosine"
cfg.num_cycles = 0.5
cfg.weight_decay = 0.0
cfg.warmup = 0.0
cfg.seed = -1

# eval
cfg.calc_metric = True

# resources
cfg.gpu = 0
cfg.drop_last = True

# model basics
cfg.mixup_p = 0.5
cfg.mixup_beta = 1.0
cfg.in_channels = 1
cfg.pretrained = False

# Dataset configuration
cfg.dataset = "robpicker.data.ds"

# data loading
cfg.batch_size = 8
cfg.batch_size_val = 16
cfg.sub_batch_size = 4
cfg.roi_size = [96, 96, 96]
cfg.train_sub_epochs = 8
cfg.val_sub_epochs = 1
cfg.pin_memory = False
cfg.num_workers = 4

cfg.prefetch_factor = 4
cfg.persistent_workers = True

# Post-processing and metrics
cfg.post_process_pipeline = "robpicker.postprocess.pp"
cfg.metric = "robpicker.metrics.metric"

# transforms
cfg.static_transforms = mt.Compose([
    mt.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    mt.NormalizeIntensityd(keys="image"),
])

cfg.train_aug = mt.Compose([
    mt.RandSpatialCropSamplesd(
        keys=["image", "label"],
        roi_size=cfg.roi_size,
        num_samples=cfg.sub_batch_size,
    ),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    mt.RandRotate90d(keys=["image", "label"], prob=0.75, max_k=3, spatial_axes=(0, 1)),
    mt.RandRotated(
        keys=["image", "label"],
        prob=0.5,
        range_x=0.78,
        range_y=0.0,
        range_z=0.0,
        padding_mode="reflection",
    ),
])

cfg.val_aug = mt.Compose([
    mt.GridPatchd(keys=["image", "label"], patch_size=cfg.roi_size, pad_mode="reflect"),
])

basic_cfg = cfg
