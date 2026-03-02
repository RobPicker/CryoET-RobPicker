import os
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

import torch
from torch import nn



def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool3d(x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)

def reconstruct(img, locations, out_size, crop_size):
    reconstructed_img = torch.zeros(out_size, device=img.device, dtype=img.dtype)

    for i in range(img.shape[0]):
        reconstructed_img[:,locations[0][i]:locations[0][i]+crop_size[0],
                          locations[1][i]:locations[1][i]+crop_size[1],
                          locations[2][i]:locations[2][i]+crop_size[2],
                         ] = img[i,:]
    return reconstructed_img

def write_mrc2(array, filename, voxel_size):
    import mrcfile
    array = array.cpu().numpy()
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(array)
        mrc.voxel_size = voxel_size

def process_single_experiment(cfg, logits, locations, experiment_name=None):
    """
    Process patches from a single experiment/tomogram.

    Args:
        cfg: Config object
        logits: Tensor of logits for patches from this experiment
        locations: Tensor of locations for patches from this experiment
        experiment_name: Name of the experiment (e.g., 'TS_5_4')

    Returns:
        DataFrame with predictions for this experiment
    """
    # Interpolate to target size
    img = torch.nn.functional.interpolate(
        logits,
        size=(cfg.roi_size[0], cfg.roi_size[1], cfg.roi_size[2]),
        mode='trilinear',
        align_corners=False
    )

    # Reconstruct full tomogram
    out_size = [cfg.n_classes + 1] + [l.item() + r for l, r in zip(locations.max(0)[0], cfg.roi_size)]
    rec_img = reconstruct(img, locations.permute(1, 0), out_size=out_size, crop_size=cfg.roi_size)

    # Use max pooling to downsample by factor of 2
    rec_img = F.max_pool3d(rec_img[None], kernel_size=2, stride=2)[0]

    # Get predictions via softmax
    preds = rec_img.softmax(0)[:-1]

    voxel_spacing = getattr(cfg, "voxel_spacing", 10.0)
    downsample_factor = 2
    effective_spacing = voxel_spacing * downsample_factor
    conf_thresh = getattr(cfg, "pp_conf_thresh", 0.01)
    x_max = getattr(cfg, "pp_x_max", 6300)
    y_max = getattr(cfg, "pp_y_max", 6300)
    z_max = getattr(cfg, "pp_z_max", 1840)

    pred_dfs = []
    for i, p in enumerate(cfg.classes):
        p1 = preds[i][None,].cuda()
        y = simple_nms(p1, nms_radius=int(0.5 * cfg.particle_radi[p] / voxel_spacing))
        kps = torch.where(y > 0)
        xyz = torch.stack(kps[1:], -1) * effective_spacing
        conf = y[kps]
        pred_df_ = pd.DataFrame(xyz.cpu().numpy(), columns=['x', 'y', 'z'])
        pred_df_['particle_type'] = p
        pred_df_['conf'] = conf.cpu().numpy()
        if experiment_name is not None:
            pred_df_['experiment'] = experiment_name
        pred_dfs.append(pred_df_)

    pred_df = pd.concat(pred_dfs)
    # Filter by bounds and confidence
    pred_df = pred_df[
        (pred_df['x'] < x_max) &
        (pred_df['y'] < y_max) &
        (pred_df['z'] < z_max) &
        (pred_df['conf'] > conf_thresh)
    ].copy()

    return pred_df


def post_process_pipeline(cfg, val_data, val_df):
    """
    Post-process validation data to extract particle predictions.

    Supports both single-experiment (legacy) and multi-experiment modes.
    Multi-experiment mode is activated when 'experiment_idx' is present in val_data.
    """
    logits = val_data['logits']
    locations = val_data['location']

    # Check if we have experiment indices (multi-experiment mode)
    has_experiment_idx = 'experiment_idx' in val_data and val_data['experiment_idx'] is not None

    if has_experiment_idx:
        # Multi-experiment mode: process each experiment separately
        experiment_indices = val_data['experiment_idx'].cpu().numpy()
        unique_exp_indices = np.unique(experiment_indices)

        # Get experiment names from dataset if available
        experiment_names = getattr(cfg, 'val_experiment_names', None)

        all_pred_dfs = []
        for exp_idx in unique_exp_indices:
            # Get mask for patches belonging to this experiment
            mask = experiment_indices == exp_idx

            # Get experiment name
            if experiment_names is not None and exp_idx < len(experiment_names):
                exp_name = experiment_names[exp_idx]
            else:
                exp_name = f"experiment_{exp_idx}"

            # Filter logits and locations for this experiment
            exp_logits = logits[mask]
            exp_locations = locations[mask]

            print(f"Processing experiment {exp_name}: {exp_logits.shape[0]} patches")

            # Process this experiment
            pred_df = process_single_experiment(cfg, exp_logits, exp_locations, exp_name)
            all_pred_dfs.append(pred_df)

        pred_df = pd.concat(all_pred_dfs, ignore_index=True)
    else:
        # Legacy single-experiment mode
        pred_df = process_single_experiment(cfg, logits, locations)
        # Try to get experiment name from val_df
        if 'experiment' in val_df.columns:
            experiments = val_df['experiment'].unique()
            if len(experiments) == 1:
                pred_df['experiment'] = experiments[0]
            else:
                raise ValueError("Multiple experiments found in val_df for single-experiment mode. Need to track experiment indices in dataloading.")

    print(f"Total predicted particles: {len(pred_df)}")
    pred_df.to_csv(os.path.join(cfg.output_dir, f"val_pred_df_seed{cfg.seed}.csv"), index=False)
    return pred_df
