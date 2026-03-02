"""
Evaluation script for inference and optional metric calculation.

Usage:
    python evaluate.py \
        --checkpoint /path/to/checkpoint_best.pth \
        --data_dir /empiar_11830_data/test \
        --output_dir ./eval_results

Optional:
    --inference_only
    --use_greedy_nms
    --thresholds 0.1,0.1
"""

import os
import json
import argparse
import importlib
from copy import copy
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import mrcfile

from monai import transforms as mt

from robpicker.data import ds


# ============================================================================
# Greedy NMS with OKS similarity (from Kaggle kernel inference_kernel.py)
# ============================================================================

def keypoint_similarity(pts1: torch.Tensor, pts2: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute OKS (Object Keypoint Similarity) between two sets of keypoints.

    Args:
        pts1: First set of points, shape [..., 3] (x, y, z)
        pts2: Second set of points, shape [..., 3] (x, y, z)
        sigma: Standard deviation for Gaussian decay (typically particle radius)

    Returns:
        Similarity scores in [0, 1], shape [...]
    """
    d = ((pts1 - pts2) ** 2).sum(dim=-1, keepdim=False)
    e = d / (2 * sigma ** 2)
    return torch.exp(-e)


@torch.no_grad()
def greedy_nms_with_oks(
    centers: torch.Tensor,
    scores: torch.Tensor,
    sigma: float,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.01,
) -> torch.Tensor:
    """
    Greedy NMS using OKS (Object Keypoint Similarity).

    Args:
        centers: Detection centers, shape [N, 3] (x, y, z)
        scores: Detection scores, shape [N]
        sigma: Sigma for OKS calculation (particle radius in voxels)
        iou_threshold: Threshold above which detections are suppressed
        score_threshold: Minimum score to consider

    Returns:
        Indices of kept detections
    """
    if len(scores) == 0:
        return torch.tensor([], dtype=torch.long, device=scores.device)

    mask = scores >= score_threshold
    if not mask.any():
        return torch.tensor([], dtype=torch.long, device=scores.device)

    valid_indices = torch.where(mask)[0]
    valid_scores = scores[mask]
    valid_centers = centers[mask]

    sorted_indices = valid_scores.argsort(descending=True)
    valid_scores = valid_scores[sorted_indices]
    valid_centers = valid_centers[sorted_indices]
    valid_indices = valid_indices[sorted_indices]

    keep_indices = []
    suppressed = torch.zeros(len(valid_scores), dtype=torch.bool, device=scores.device)

    for i in range(len(valid_scores)):
        if suppressed[i]:
            continue
        keep_indices.append(valid_indices[i].item())

        oks = keypoint_similarity(valid_centers[i:i + 1, :], valid_centers, sigma)
        suppressed |= oks > iou_threshold

    return torch.tensor(keep_indices, dtype=torch.long, device=scores.device)


def load_annotations(data_dir, classes, class_mapping, expected_spacing=None):
    all_annotations = []
    tomograms = ds.discover_tomograms(data_dir, expected_spacing=expected_spacing)
    for tomo in tqdm(tomograms, desc="Loading annotations"):
        tomo_name = tomo['tomo_name']
        voxel_spacing = tomo.get('voxel_spacing', 10.0)
        annotations = ds.load_annotations(tomo['xml_path'], classes, class_mapping)
        for particle_type, coords in annotations.items():
            for x, y, z in coords:
                all_annotations.append({
                    'experiment': str(tomo_name),
                    'x': x * voxel_spacing,
                    'y': y * voxel_spacing,
                    'z': z * voxel_spacing,
                    'particle_type': particle_type,
                })
    if not all_annotations:
        return pd.DataFrame(columns=['experiment', 'x', 'y', 'z', 'particle_type'])
    return pd.DataFrame(all_annotations)


class EvalDataset(Dataset):
    """Dataset for inference on EMPIAR-style tomograms."""

    def __init__(self, cfg, data_dir, patch_overlap=0, preload_all=True):
        self.cfg = cfg
        self.data_dir = data_dir
        self.roi_size = cfg.roi_size
        self.patch_overlap = patch_overlap
        self.preload_all = preload_all

        self.tomograms = ds.discover_tomograms(data_dir, expected_spacing=cfg.voxel_spacing)
        self.experiments = [str(t['tomo_name']) for t in self.tomograms]

        self.tomogram_info = {}
        for t in self.tomograms:
            tomo_name = str(t['tomo_name'])
            self.tomogram_info[tomo_name] = {
                'path': t['mrc_path'],
                'voxel_spacing': t.get('voxel_spacing', 10.0),
                'tomo_size': t.get('tomo_size'),
            }
        if self.tomogram_info:
            self.actual_voxel_spacing = list(self.tomogram_info.values())[0]['voxel_spacing']

        self.static_transforms = mt.Compose([
            mt.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            mt.NormalizeIntensityd(keys="image"),
        ])
        self.val_aug = mt.Compose([
            mt.GridPatchd(
                keys=["image"],
                patch_size=self.roi_size,
                pad_mode='reflect',
                overlap=self.patch_overlap,
            )
        ])

        self.processed_data = {}
        self.patch_indices = []
        if self.preload_all:
            print("Loading and preprocessing tomograms with MONAI transforms...")
            for exp in tqdm(self.experiments, desc="Preprocessing"):
                self.processed_data[exp] = self._process_experiment(exp)
            for exp in self.experiments:
                n_patches = self.processed_data[exp]['n_patches']
                for patch_idx in range(n_patches):
                    self.patch_indices.append((exp, patch_idx))
            print(f"Total patches: {len(self.patch_indices)}")

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        exp, patch_idx = self.patch_indices[idx]
        data = self.processed_data[exp]

        image_tensor = data['image'][patch_idx]
        loc_data = data['image'].meta['location'][:, patch_idx]
        if isinstance(loc_data, np.ndarray):
            location = torch.from_numpy(loc_data)
        else:
            location = loc_data.clone()

        return {
            'experiment': exp,
            'input': image_tensor,
            'location': location,
        }

    def _process_experiment(self, exp):
        info = self.tomogram_info[exp]
        with mrcfile.open(info['path'], mode='r', permissive=True) as mrc:
            img = mrc.data.copy().astype(np.float32)
        img = img.transpose(2, 1, 0)

        sample = {'image': img}
        sample = self.static_transforms(sample)
        sample = self.val_aug(sample)

        import monai.data as md
        monai_ds = md.CacheDataset(data=[sample], transform=None, cache_rate=1.0, progress=False)[0]
        return {
            'image': monai_ds['image'],
            'n_patches': len(monai_ds['image']),
        }


def reconstruct_avg(img, locations, out_size, crop_size):
    """Reconstruct volume by averaging logits over overlapping patches."""
    reconstructed_img = torch.zeros(out_size, device=img.device, dtype=img.dtype)
    counts = torch.zeros(out_size[1:], device=img.device, dtype=img.dtype)

    for i in range(img.shape[0]):
        xs = locations[0][i]
        ys = locations[1][i]
        zs = locations[2][i]
        reconstructed_img[:,
                          xs:xs + crop_size[0],
                          ys:ys + crop_size[1],
                          zs:zs + crop_size[2]] += img[i, :]
        counts[xs:xs + crop_size[0], ys:ys + crop_size[1], zs:zs + crop_size[2]] += 1

    counts = counts.clamp_min(1.0)
    reconstructed_img = reconstructed_img / counts.unsqueeze(0)
    return reconstructed_img


def normalize_patch_overlap(patch_overlap, roi_size):
    """Normalize patch overlap to MONAI's expected ratio format."""
    if isinstance(patch_overlap, str) and "," in patch_overlap:
        parts = [p.strip() for p in patch_overlap.split(",")]
        values = tuple(float(p) for p in parts)
    elif isinstance(patch_overlap, (tuple, list)):
        values = tuple(float(v) for v in patch_overlap)
    else:
        values = float(patch_overlap)

    if isinstance(values, tuple):
        if len(values) != 3:
            raise ValueError(f"patch_overlap must have 3 values, got {values}")
        overlap_ratios = []
        for v, size in zip(values, roi_size):
            ratio = v / size if v >= 1.0 else v
            if ratio < 0 or ratio >= 1:
                raise ValueError(f"patch_overlap ratio must be in [0, 1), got {ratio}")
            overlap_ratios.append(ratio)
        overlap = tuple(overlap_ratios)
    else:
        v = values
        ratio = v / float(roi_size[0]) if v >= 1.0 else v
        if ratio < 0 or ratio >= 1:
            raise ValueError(f"patch_overlap ratio must be in [0, 1), got {ratio}")
        overlap = ratio

    enabled = False
    if isinstance(overlap, tuple):
        enabled = any(v > 0 for v in overlap)
    else:
        enabled = overlap > 0
    return overlap, enabled


def run_inference_and_postprocess(
    model,
    dataset,
    cfg,
    device,
    use_greedy_nms=False,
    iou_threshold=0.5,
    enforce_unique_class=False,
    flip_tta=False,
):
    """
    Run inference and post-processing per experiment.

    Supports two NMS methods:
    1. Simple NMS (default): Uses max_pool3d like pp.py
    2. Greedy NMS with OKS (--use_greedy_nms)
    """
    from robpicker.postprocess.pp import simple_nms, reconstruct

    model.eval()
    all_predictions = []

    voxel_spacing = getattr(dataset, 'actual_voxel_spacing', 10.0)
    effective_voxel_spacing = voxel_spacing * 2
    print(f"Using voxel spacing: {voxel_spacing:.3f}A (effective: {effective_voxel_spacing:.3f}A)")

    x_max = getattr(cfg, 'pp_x_max', 6300)
    y_max = getattr(cfg, 'pp_y_max', 6300)
    z_max = getattr(cfg, 'pp_z_max', 1840)
    conf_thresh = getattr(cfg, 'pp_conf_thresh', 0.01)

    with torch.no_grad():
        for exp in tqdm(dataset.experiments, desc="Processing experiments"):
            if dataset.preload_all:
                patch_indices = [
                    idx for idx, (e, _) in enumerate(dataset.patch_indices) if e == exp
                ]
                get_item = dataset.__getitem__
            else:
                data_entry = dataset._process_experiment(exp)
                image_tensor = data_entry['image']
                n_patches = data_entry['n_patches']

            all_logits = []
            all_locations = []

            batch_size = cfg.batch_size_val or 16
            if dataset.preload_all:
                total_patches = len(patch_indices)
            else:
                total_patches = n_patches

            for i in range(0, total_patches, batch_size):
                if dataset.preload_all:
                    batch_indices = patch_indices[i:i + batch_size]
                    batch_inputs = []
                    batch_locs = []
                    for idx in batch_indices:
                        item = get_item(idx)
                        batch_inputs.append(item['input'])
                        batch_locs.append(item['location'])
                    inputs = torch.stack(batch_inputs, dim=0).to(device)
                    locations = torch.stack(batch_locs, dim=0)
                else:
                    batch_indices = list(range(i, min(i + batch_size, total_patches)))
                    inputs = image_tensor[batch_indices].to(device)
                    loc_data = image_tensor.meta['location'][:, batch_indices]
                    if isinstance(loc_data, np.ndarray):
                        locations = torch.from_numpy(loc_data)
                    else:
                        locations = loc_data.clone()
                    locations = locations.permute(1, 0)

                if flip_tta:
                    flip_dims = (2, 3, 4)
                    outputs = model({'input': inputs})
                    logits = outputs['logits']
                    tta_inputs = torch.flip(inputs, dims=flip_dims)
                    tta_outputs = model({'input': tta_inputs})
                    tta_logits = torch.flip(tta_outputs['logits'], dims=flip_dims)
                    logits = (logits + tta_logits) / 2.0
                else:
                    outputs = model({'input': inputs})
                    logits = outputs['logits']

                all_logits.append(logits)
                all_locations.append(locations)

            logits = torch.cat(all_logits, dim=0)
            locations = torch.cat(all_locations, dim=0)

            img = F.interpolate(logits, size=tuple(cfg.roi_size), mode='trilinear', align_corners=False)

            out_size = [cfg.n_classes + 1] + [
                int(l.item()) + r for l, r in zip(locations.max(0)[0], cfg.roi_size)
            ]

            if getattr(dataset, "patch_overlap_enabled", False):
                rec_img = reconstruct_avg(img, locations.permute(1, 0), out_size=out_size, crop_size=cfg.roi_size)
            else:
                rec_img = reconstruct(img, locations.permute(1, 0), out_size=out_size, crop_size=cfg.roi_size)

            s = rec_img.shape[-3:]
            rec_img = F.interpolate(
                rec_img[None],
                size=(s[0] // 2, s[1] // 2, s[2] // 2),
                mode='trilinear',
                align_corners=False
            )[0]

            preds = rec_img.softmax(0)[:-1]
            if enforce_unique_class:
                class_max = preds.argmax(0, keepdim=True)
                class_mask = torch.zeros_like(preds, dtype=torch.bool)
                class_mask.scatter_(0, class_max, True)
                preds = preds * class_mask

            for class_idx, class_name in enumerate(cfg.classes):
                p1 = preds[class_idx][None,].to(device)
                sigma_voxels = cfg.particle_radi[class_name] / effective_voxel_spacing

                if use_greedy_nms:
                    score_threshold = conf_thresh
                    scores_flat = p1.flatten()
                    indices = torch.arange(scores_flat.numel(), device=device)
                    mask = scores_flat >= score_threshold
                    valid_scores = scores_flat[mask]
                    valid_indices = indices[mask]

                    if len(valid_scores) > 0:
                        shape = p1.shape[1:]
                        d_idx = valid_indices // (shape[1] * shape[2])
                        hw_idx = valid_indices % (shape[1] * shape[2])
                        h_idx = hw_idx // shape[2]
                        w_idx = hw_idx % shape[2]
                        centers_voxels = torch.stack([d_idx, h_idx, w_idx], dim=-1).float()

                        keep_indices = greedy_nms_with_oks(
                            centers=centers_voxels,
                            scores=valid_scores,
                            sigma=sigma_voxels,
                            iou_threshold=iou_threshold,
                            score_threshold=score_threshold,
                        )

                        xyz_voxels = centers_voxels[keep_indices]
                        conf = valid_scores[keep_indices]
                        xyz = xyz_voxels * effective_voxel_spacing

                        for j in range(len(xyz)):
                            x, y_coord, z = xyz[j].cpu().numpy()
                            c = conf[j].cpu().item()
                            if x < x_max and y_coord < y_max and z < z_max:
                                all_predictions.append({
                                    'experiment': exp,
                                    'x': float(x),
                                    'y': float(y_coord),
                                    'z': float(z),
                                    'particle_type': class_name,
                                    'conf': c,
                                })
                else:
                    nms_radius = int(sigma_voxels)
                    y = simple_nms(p1, nms_radius=max(1, nms_radius))

                    kps = torch.where(y > 0)
                    xyz = torch.stack(kps[1:], -1) * effective_voxel_spacing
                    conf = y[kps]

                    for j in range(len(xyz)):
                        x, y_coord, z = xyz[j].cpu().numpy()
                        c = conf[j].cpu().item()
                        if x < x_max and y_coord < y_max and z < z_max and c > conf_thresh:
                            all_predictions.append({
                                'experiment': exp,
                                'x': float(x),
                                'y': float(y_coord),
                                'z': float(z),
                                'particle_type': class_name,
                                'conf': c,
                            })

    return all_predictions


def resolve_config_module(name: str) -> str:
    """Resolve config module path for importlib."""
    if "." in name:
        return name
    return f"robpicker.configs.{name}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--data_dir", required=True, help="Path to test data directory")
    parser.add_argument("--output_dir", default="./eval_results", help="Output directory")
    parser.add_argument("--config", default="cfg_resnet34", help="Config file name (without .py)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--patch_overlap", default=0.3,
                        help="Patch overlap for extracting patches from tomogram to run inference on. Use int/float or 'x,y,z' (voxels if >=1, ratio if <1).")
    parser.add_argument("--gt_csv", default=None,
                        help="Optional: path to ground truth CSV file containing annotation columns: x,y,z,particle_type,experiment.")
    parser.add_argument("--use_greedy_nms", action="store_true", default=True,
                        help="Use greedy non-maximum suppression (NMS) with OKS similarity instead of simple max_pool3d NMS.")
    parser.add_argument("--no_greedy_nms", dest="use_greedy_nms", action="store_false",
                        help="Disable greedy NMS and use simple max_pool3d non-maximum suppression (NMS).")
    parser.add_argument("--iou_threshold", type=float, default=0.8,
                        help="OKS threshold for greedy NMS (default: 0.8).")
    parser.add_argument("--thresholds", default=None,
                        help="Optional per-class thresholds (comma-separated) in cfg.classes order.")
    parser.add_argument("--threshold_range", default="0.1,0.6,0.005",
                        help="Grid search range as start,end,step (default: 0.1,0.6,0.005).")
    parser.add_argument("--enforce_unique_class", action="store_true",
                        help="Keep only the max class per voxel before NMS to avoid cross-class duplicates.")
    parser.add_argument("--flip_tta", action="store_true", default=True,
                        help="Use flip TTA (average original and all-axes flipped predictions).")
    parser.add_argument("--no_flip_tta", dest="flip_tta", action="store_false",
                        help="Disable flip TTA.")
    parser.add_argument("--stream_experiments", action="store_true", default=True,
                        help="Load and process one experiment at a time to reduce memory usage.")
    parser.add_argument("--no_stream_experiments", dest="stream_experiments", action="store_false",
                        help="Preload all experiments into memory before inference.")
    parser.add_argument("--inference_only", action="store_true",
                        help="Run inference only; skip metric computation.")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading config: {args.config}")
    cfg = copy(importlib.import_module(resolve_config_module(args.config)).cfg)
    cfg.device = device
    cfg.batch_size_val = args.batch_size

    patch_overlap, patch_overlap_enabled = normalize_patch_overlap(args.patch_overlap, cfg.roi_size)

    print(f"Scanning data directory: {args.data_dir}")
    tomograms = ds.discover_tomograms(args.data_dir, expected_spacing=cfg.voxel_spacing)
    experiments = [str(t['tomo_name']) for t in tomograms]
    print(f"Found {len(experiments)} tomograms")

    if args.inference_only:
        gt_df = pd.DataFrame(columns=['experiment', 'x', 'y', 'z', 'particle_type'])
    elif args.gt_csv:
        print(f"Loading ground truth from CSV: {args.gt_csv}")
        gt_df = pd.read_csv(args.gt_csv)
        gt_df = gt_df[gt_df['experiment'].isin(experiments)].copy()
    else:
        print("Loading ground truth annotations from XML...")
        gt_df = load_annotations(
            args.data_dir,
            cfg.classes,
            cfg.class_mapping,
            expected_spacing=cfg.voxel_spacing,
        )
    if not args.inference_only:
        print(f"Loaded {len(gt_df)} ground truth annotations")
        if len(gt_df) > 0:
            print(f"Particle type distribution:\n{gt_df['particle_type'].value_counts()}")

    print("Creating dataset...")
    dataset = EvalDataset(
        cfg,
        args.data_dir,
        patch_overlap=patch_overlap,
        preload_all=not args.stream_experiments,
    )
    dataset.patch_overlap_enabled = patch_overlap_enabled

    print(f"Loading model from: {args.checkpoint}")
    Net = importlib.import_module(cfg.model).Net
    model = Net(cfg)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'main' in checkpoint:
        state_dict = checkpoint['main']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    nms_method = "greedy NMS with OKS" if args.use_greedy_nms else "simple NMS (max_pool3d)"
    print("Running inference and post-processing...")
    print(f"  NMS method: {nms_method}")
    if args.use_greedy_nms:
        print(f"  OKS threshold: {args.iou_threshold}")
    all_predictions = run_inference_and_postprocess(
        model, dataset, cfg, device,
        use_greedy_nms=args.use_greedy_nms,
        iou_threshold=args.iou_threshold,
        enforce_unique_class=args.enforce_unique_class,
        flip_tta=args.flip_tta,
    )

    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
    else:
        pred_df = pd.DataFrame(columns=['experiment', 'x', 'y', 'z', 'particle_type', 'conf'])

    print(f"Total predictions: {len(pred_df)}")
    if len(pred_df) > 0:
        print(f"Prediction distribution:\n{pred_df['particle_type'].value_counts()}")

    pred_path = os.path.join(args.output_dir, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path}")

    if args.inference_only:
        if args.thresholds is not None:
            parts = [p.strip() for p in str(args.thresholds).split(",") if p.strip()]
            if len(parts) != len(cfg.classes):
                raise ValueError(
                    f"--thresholds expects {len(cfg.classes)} values, got {len(parts)}"
                )
            best_ths = {cls: float(th) for cls, th in zip(cfg.classes, parts)}
            print(f"Applying provided thresholds in inference-only mode: {best_ths}")
            submission_pp = []
            for p in cfg.classes:
                th = best_ths[p]
                submission_pp.append(pred_df[(pred_df['particle_type'] == p) & (pred_df['conf'] > th)])
            pred_pp = pd.concat(submission_pp) if submission_pp else pd.DataFrame(
                columns=['experiment', 'x', 'y', 'z', 'particle_type', 'conf']
            )
            pred_pp_path = os.path.join(args.output_dir, 'predictions_thresholded.csv')
            pred_pp.to_csv(pred_pp_path, index=False)
            print(f"Saved thresholded predictions to: {pred_pp_path}")
        print("Inference-only mode: skipping metric calculation")
        return

    if len(gt_df) == 0 or len(pred_df) == 0:
        print("WARNING: No predictions or ground truth available for metric calculation")
        return

    from robpicker.metrics.metric import score as compute_fbeta_score

    solution = gt_df.copy()
    solution['id'] = range(len(solution))

    submission = pred_df.copy()
    submission['id'] = range(len(submission))

    best_ths = {}
    if args.thresholds is not None:
        parts = [p.strip() for p in str(args.thresholds).split(",") if p.strip()]
        if len(parts) != len(cfg.classes):
            raise ValueError(
                f"--thresholds expects {len(cfg.classes)} values, got {len(parts)}"
            )
        best_ths = {cls: float(th) for cls, th in zip(cfg.classes, parts)}
        print(f"Using provided thresholds: {best_ths}")
    else:
        print("Finding optimal thresholds per particle type...")
        parts = [p.strip() for p in str(args.threshold_range).split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("--threshold_range expects start,end,step")
        th_start, th_end, th_step = (float(x) for x in parts)
        for p in tqdm(cfg.classes, desc="Optimizing thresholds"):
            sol_p = solution[solution['particle_type'] == p].copy()
            sub_p = submission[submission['particle_type'] == p].copy()

            if len(sol_p) == 0:
                best_ths[p] = 0.0
                continue

            best_score = -1
            best_th = 0.0
            for th in np.arange(th_start, th_end, th_step):
                sub_filtered = sub_p[sub_p['conf'] > th].copy()
                if len(sub_filtered) == 0:
                    continue
                try:
                    s, _ = compute_fbeta_score(
                        sol_p,
                        sub_filtered,
                        row_id_column_name='id',
                        distance_multiplier=getattr(cfg, "metric_distance_multiplier", 0.5),
                        beta=getattr(cfg, "metric_beta", 1),
                        weighted=False,
                        cfg=cfg,
                    )
                    if s > best_score:
                        best_score = s
                        best_th = th
                except Exception:
                    continue
            best_ths[p] = best_th

    submission_pp = []
    for p in cfg.classes:
        th = best_ths[p]
        submission_pp.append(submission[(submission['particle_type'] == p) & (submission['conf'] > th)])
    submission_pp = pd.concat(submission_pp) if submission_pp else pd.DataFrame()

    print("Computing final scores...")
    score_total, particle_scores = compute_fbeta_score(
        solution,
        submission_pp,
        row_id_column_name='id',
        distance_multiplier=getattr(cfg, "metric_distance_multiplier", 0.5),
        beta=getattr(cfg, "metric_beta", 1),
        weighted=True,
        cfg=cfg,
    )

    scores = {f'score_{p}': particle_scores.get(p, 0.0) for p in cfg.classes}
    scores['score'] = score_total
    scores['best_thresholds'] = best_ths

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nScore: {score_total:.4f}")
    print("\n" + "-" * 60)
    print("Per-particle F-beta scores:")
    print("-" * 60)
    print(f"  {'Particle Type':<25} {'F-beta':>10} {'Threshold':>10}")
    print("-" * 60)
    for class_name in cfg.classes:
        fbeta = particle_scores.get(class_name, 0.0)
        th = best_ths.get(class_name, 0.0)
        print(f"  {class_name:<25} {fbeta:>10.4f} {th:>10.3f}")
    print("=" * 60)

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(scores, f, indent=2, default=float)
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
