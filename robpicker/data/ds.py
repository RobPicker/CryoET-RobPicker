"""
Dataset for EMPIAR-style format (MRC + XML).

This dataset reads tomograms and annotations from an EMPIAR-style directory structure
with train/, meta/, test/ folders containing MRC files and XML annotations.

Data format:
- MRC files: {tomo_name}.mrc (e.g., tomo0001.mrc or 1287.mrc)
- XML files: {tomo_name}_objl.xml (e.g., tomo0001_objl.xml)
- XML annotations contain: <object tomo_name="tomo0001" class_label="1" x="592" y="772" z="287" .../>
- Class labels: 1 and 2 (two ribosome classes)

Usage:
    # In config file:
    cfg.dataset = "robpicker.data.ds"
    cfg.data_dir = "/empiar_11830_data"
    cfg.train_folder = "train"  # for main training
    cfg.meta_folder = "meta"    # for meta-learning
    cfg.test_folder = "test"    # for validation/testing
"""

import os
import re
import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import mrcfile
from tqdm import tqdm
from glob import glob

import monai.data as md
import monai.transforms as mt


class ClassAwareRandCropSamplesd(mt.MapTransform):
    """Class-aware random crop with sampling bias toward specified classes."""

    def __init__(self, keys, roi_size, num_samples, class_weights, bg_weight=0.1):
        super().__init__(keys)
        self.roi_size = tuple(roi_size)
        self.num_samples = int(num_samples)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.bg_weight = float(bg_weight)

    def _sample_center(self, label):
        spatial = label.shape[-3:]
        weights = torch.cat([self.class_weights, torch.tensor([self.bg_weight])])
        probs = weights / weights.sum()
        cls = int(torch.multinomial(probs, 1).item())
        if cls == label.shape[0]:
            return [int(torch.randint(0, s, (1,)).item()) for s in spatial]

        coords = torch.nonzero(label[cls] > 0, as_tuple=False)
        if coords.numel() == 0:
            return [int(torch.randint(0, s, (1,)).item()) for s in spatial]
        idx = int(torch.randint(0, coords.shape[0], (1,)).item())
        return coords[idx].tolist()

    def _sample_start(self, spatial, point):
        starts = []
        for p, rs, s in zip(point, self.roi_size, spatial):
            min_start = max(0, p - (rs - 1))
            max_start = min(p, s - rs)
            if max_start < min_start:
                start = max(0, min(p - rs // 2, s - rs))
            else:
                start = int(torch.randint(min_start, max_start + 1, (1,)).item())
            starts.append(start)
        return starts

    def _crop(self, img, start):
        xs, ys, zs = start
        xe, ye, ze = xs + self.roi_size[0], ys + self.roi_size[1], zs + self.roi_size[2]
        return img[..., xs:xe, ys:ye, zs:ze]

    def __call__(self, data):
        d = dict(data)
        img = d["image"]
        label = d["label"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        samples = []
        for _ in range(self.num_samples):
            point = self._sample_center(label)
            start = self._sample_start(label.shape[-3:], point)
            sample = dict(d)
            for key in self.keys:
                if key == "image":
                    sample[key] = self._crop(img, start)
                elif key == "label":
                    sample[key] = self._crop(label, start)
            samples.append(sample)
        return samples


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def collate_fn(batch):
    keys = batch[0].keys()
    batch_dict = {key: torch.cat([b[key] for b in batch]) for key in keys}
    return batch_dict


tr_collate_fn = collate_fn
val_collate_fn = collate_fn


def discover_tomograms(data_folder: str, expected_spacing: float | None = None) -> list[dict]:
    """
    Discover all tomograms in an EMPIAR-style directory.

    Args:
        data_folder: Path to split folder (e.g., /empiar_11830_data/train)

    Returns:
        List of dicts with tomogram info: {
            'tomo_name': str,     # e.g., 'tomo0001' or '1287'
            'mrc_path': str,      # path to MRC file
            'xml_path': str,      # path to XML annotation file
            'tomo_size': dict,    # {'x': int, 'y': int, 'z': int}
            'voxel_spacing': float
        }
    """
    tomograms = []

    # Find all MRC files (excluding *_target_*.mrc files)
    mrc_files = glob(os.path.join(data_folder, '*.mrc'))

    for mrc_path in sorted(mrc_files):
        mrc_name = os.path.basename(mrc_path)
        tomo_name = mrc_name.replace('.mrc', '')

        # Look for corresponding XML file
        # Preferred naming: {tomo_name}_objl.xml
        # Backward compatibility: tomo{num}_objl.xml with optional zero-padding
        xml_candidates = [
            os.path.join(data_folder, f'{tomo_name}_objl.xml'),
        ]
        if not tomo_name.startswith("tomo"):
            xml_candidates.append(os.path.join(data_folder, f'tomo{tomo_name}_objl.xml'))
        if tomo_name.isdigit():
            xml_candidates.append(os.path.join(data_folder, f'tomo{tomo_name.zfill(4)}_objl.xml'))
            xml_candidates.append(os.path.join(data_folder, f'tomo{int(tomo_name):04d}_objl.xml'))

        xml_path = None
        for xml_candidate in xml_candidates:
            if os.path.exists(xml_candidate):
                xml_path = xml_candidate
                break

        if xml_path is None:
            print(f"Warning: No XML annotation found for {mrc_name}")
            continue

        # Get tomogram size from MRC file
        try:
            with mrcfile.open(mrc_path, permissive=True) as mrc:
                # MRC shape is (Z, Y, X)
                shape = mrc.data.shape
                voxel_size = float(mrc.voxel_size.x)
                if expected_spacing is not None:
                    expected_spacing = float(expected_spacing)
                    if not np.isclose(voxel_size, expected_spacing, rtol=5e-2, atol=2e-1):
                        raise ValueError(
                            f"Voxel spacing mismatch for {mrc_name}: "
                            f"MRC={voxel_size}, expected={expected_spacing}"
                        )
                tomo_size = {
                    'x': shape[2],  # X
                    'y': shape[1],  # Y
                    'z': shape[0],  # Z
                }
        except Exception as e:
            print(f"Warning: Could not read MRC {mrc_path}: {e}")
            continue

        tomograms.append({
            'tomo_name': tomo_name,
            'mrc_path': mrc_path,
            'xml_path': xml_path,
            'tomo_size': tomo_size,
            'voxel_spacing': float(voxel_size),
        })

    return tomograms


def load_annotations(xml_path: str, classes: list[str], class_mapping: dict[int, str]) -> dict:
    """
    Load annotations from EMPIAR-style XML format.

    Args:
        xml_path: Path to XML annotation file
        classes: List of particle class names

    Returns:
        Dict mapping class name to list of (x, y, z) coordinates in voxels
    """
    annotations = {cls: [] for cls in classes}

    if xml_path is None or not os.path.exists(xml_path):
        return annotations

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_label = int(obj.get('class_label'))
            x = float(obj.get('x'))
            y = float(obj.get('y'))
            z = float(obj.get('z'))

            # Map class label to class name
            class_name = class_mapping.get(class_label)
            if class_name is not None and class_name in classes:
                annotations[class_name].append((x, y, z))

    except Exception as e:
        print(f"Warning: Could not parse XML {xml_path}: {e}")

    return annotations


class CustomDataset(Dataset):
    """
    Dataset for EMPIAR-style format.

    Automatically discovers all tomograms in the data_folder and loads
    annotations from XML files.
    """

    def __init__(self, df, cfg, aug, mode="train"):
        """
        Args:
            df: Ignored (kept for interface compatibility). Can be None.
            cfg: Config object with required fields:
                - classes: list of particle class names
                - data_dir: root path to EMPIAR data
                - train_folder: folder name for training data
                - meta_folder: folder name for meta data
                - test_folder: folder name for test data
            aug: MONAI transforms for augmentation
            mode: 'train', 'meta', or 'val'/'test'
        """
        self.cfg = cfg
        self.mode = mode
        self.class2id = {c: i for i, c in enumerate(cfg.classes)}
        self.n_classes = len(cfg.classes)
        self.random_transforms = aug
        self.class_mapping = getattr(cfg, "class_mapping", {})
        if not self.class_mapping:
            raise ValueError("cfg.class_mapping is required for ds (maps label IDs to class names).")

        # Determine data folder based on mode
        data_dir = getattr(cfg, 'data_dir', None)

        if mode == 'train':
            folder_name = getattr(cfg, 'train_folder', 'train')
        elif mode == 'meta':
            folder_name = getattr(cfg, 'meta_folder', 'meta')
        else:  # val or test
            folder_name = getattr(cfg, 'test_folder', 'test')

        self.data_folder = os.path.join(data_dir, folder_name)

        # Discover all tomograms
        print(f"Discovering tomograms in {self.data_folder}...")
        self.tomograms = discover_tomograms(self.data_folder, expected_spacing=cfg.voxel_spacing)
        print(f"Found {len(self.tomograms)} tomograms")

        # For validation mode, optionally limit the number of tomograms to load
        if self.mode not in ['train', 'meta']:
            val_tomogram_limit = getattr(cfg, 'val_tomogram_limit', None)
            if val_tomogram_limit is not None and val_tomogram_limit > 0:
                self.tomograms = self.tomograms[:val_tomogram_limit]
                print(f"Limiting validation to {len(self.tomograms)} tomogram(s)")

        # Build DataFrame from annotations
        df_rows = []
        for tomo_info in self.tomograms:
            tomo_name = tomo_info['tomo_name']
            voxel_spacing = tomo_info['voxel_spacing']

            # Load annotations for this tomogram
            annotations = load_annotations(tomo_info['xml_path'], cfg.classes, self.class_mapping)

            for particle_type, coords in annotations.items():
                for x, y, z in coords:
                    # Convert from voxels to angstroms
                    df_rows.append({
                        'x': x * voxel_spacing,
                        'y': y * voxel_spacing,
                        'z': z * voxel_spacing,
                        'particle_type': str(particle_type),
                        'experiment': str(tomo_name),
                        'fold': 0,  # Default fold
                    })

        self.df = pd.DataFrame(df_rows)
        print(f"Built DataFrame with {len(self.df)} annotations")
        if len(self.df) > 0:
            print(f"  Class distribution: {self.df.groupby('particle_type').size().to_dict()}")

        # Load data
        data = [self.load_one(tomo_info) for tomo_info in tqdm(self.tomograms)]
        data = md.CacheDataset(data=data, transform=cfg.static_transforms, cache_rate=1.0)

        if self.mode in ['train', 'meta']:
            resample_weight = getattr(cfg, 'resample_weight', None)
            resample_bg_weight = getattr(cfg, 'resample_bg_weight', None)
            if resample_weight is not None:
                if len(resample_weight) != self.n_classes:
                    raise ValueError(
                        f"cfg.resample_weight must have {self.n_classes} values, got {len(resample_weight)}"
                    )
                if resample_bg_weight is None:
                    min_weight = float(np.min(resample_weight))
                    resample_bg_weight = max(min_weight * 0.1, 1e-3)
                self.random_transforms = self._make_class_aware_transforms(
                    aug,
                    resample_weight,
                    resample_bg_weight,
                )

            self.monai_ds = md.Dataset(data=data, transform=self.random_transforms)
            self.sub_epochs = cfg.train_sub_epochs if mode == 'train' else getattr(cfg, 'meta_sub_epochs', cfg.train_sub_epochs)
            self.len = len(self.monai_ds) * self.sub_epochs
        else:
            # For validation, process all tomograms and concatenate their patches
            self.sub_epochs = cfg.val_sub_epochs
            val_data = md.CacheDataset(data=data, transform=self.random_transforms, cache_rate=1.0)

            # Concatenate patches from all tomograms with index tracking
            all_images = []
            all_labels = []
            all_locations = []
            all_tomo_indices = []  # Track which tomogram each patch belongs to
            self.val_experiment_names = []

            for i in range(len(val_data)):
                tomo_data = val_data[i]
                num_patches = tomo_data['image'].shape[0]

                all_images.append(tomo_data['image'])
                all_labels.append(tomo_data['label'])
                # Get location metadata from the image tensor
                if hasattr(tomo_data['image'], 'meta') and 'location' in tomo_data['image'].meta:
                    all_locations.append(tomo_data['image'].meta['location'])

                # Track tomo index for each patch
                all_tomo_indices.extend([i] * num_patches)
                self.val_experiment_names.append(str(self.tomograms[i]['tomo_name']))

            # Stack all patches
            self.val_images = torch.cat(all_images, dim=0)
            self.val_labels = torch.cat(all_labels, dim=0)
            if all_locations:
                self.val_locations = np.concatenate(all_locations, axis=1)
            else:
                self.val_locations = None

            # Store tomo indices for each patch
            self.val_tomo_indices = np.array(all_tomo_indices)
            # Store tomo names for mapping indices to names
            self.val_tomo_names = [t['tomo_name'] for t in self.tomograms]

            self.len = len(self.val_images)
            print(f"Validation dataset: {len(val_data)} tomograms, {self.len} total patches")

    def __getitem__(self, idx):
        if self.mode in ['train', 'meta']:
            monai_dict = self.monai_ds[idx // self.sub_epochs]
            feature_dict = {
                "input": torch.stack([item['image'] for item in monai_dict]),
                "target": torch.stack([item['label'] for item in monai_dict]),
            }
        else:
            # Use pre-concatenated validation data
            image = self.val_images[idx]
            label = self.val_labels[idx]

            if self.val_locations is not None:
                location = torch.from_numpy(self.val_locations[:, idx])
            else:
                location = torch.zeros(3)

            # Get tomo index for this patch
            tomo_idx = self.val_tomo_indices[idx]

            feature_dict = {
                "input": image.unsqueeze(0),
                "location": location.unsqueeze(0),
                "target": label.unsqueeze(0),
                "experiment_idx": torch.tensor([tomo_idx], dtype=torch.long),
            }

        return feature_dict

    def __len__(self):
        return self.len

    def load_one(self, tomo_info: dict) -> dict:
        """
        Load a single tomogram and its annotations.

        Args:
            tomo_info: Dict from discover_tomograms()

        Returns:
            Dict with 'image' and 'label' arrays
        """
        xml_path = tomo_info['xml_path']
        mrc_path = tomo_info['mrc_path']
        tomo_name = tomo_info['tomo_name']

        # Load from MRC file (need to transpose)
        try:
            with mrcfile.open(mrc_path, mode='r', permissive=True) as mrc:
                # MRC data has shape (Z, Y, X)
                # Transpose to (X, Y, Z) to match training format
                img = mrc.data.copy().transpose(2, 1, 0)
        except Exception as e:
            print(f"Error loading MRC {mrc_path}: {e}")
            raise

        # Load annotations
        annotations = load_annotations(xml_path, self.cfg.classes, self.class_mapping)

        # Create mask
        mask = np.zeros((self.n_classes,) + img.shape[-3:], dtype=np.float32)

        for cls_name, coords in annotations.items():
            cls_id = self.class2id[cls_name]
            for x, y, z in coords:
                # Ensure coordinates are within bounds
                xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
                if 0 <= xi < img.shape[0] and 0 <= yi < img.shape[1] and 0 <= zi < img.shape[2]:
                    mask[cls_id, xi, yi, zi] = 1

        return {'image': img, 'label': mask}

    def get_tomo_list(self) -> list[str]:
        """Return list of tomogram IDs."""
        return [t['tomo_name'] for t in self.tomograms]

    def _make_class_aware_transforms(self, aug, resample_weight, resample_bg_weight):
        """Replace RandSpatialCropSamplesd with class-aware cropping."""
        if isinstance(aug, mt.Compose):
            transforms = list(aug.transforms)
        else:
            transforms = [aug] if aug is not None else []

        replaced = False
        for i, t in enumerate(transforms):
            if isinstance(t, mt.RandSpatialCropSamplesd):
                transforms[i] = ClassAwareRandCropSamplesd(
                    keys=t.keys,
                    roi_size=getattr(t, "roi_size", self.cfg.roi_size),
                    num_samples=getattr(t, "num_samples", self.cfg.sub_batch_size),
                    class_weights=resample_weight,
                    bg_weight=resample_bg_weight,
                )
                replaced = True
                break

        if not replaced:
            transforms.insert(0, ClassAwareRandCropSamplesd(
                keys=["image", "label"],
                roi_size=self.cfg.roi_size,
                num_samples=self.cfg.sub_batch_size,
                class_weights=resample_weight,
                bg_weight=resample_bg_weight,
            ))

        return mt.Compose(transforms)
