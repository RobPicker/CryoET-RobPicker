# Inference and Evaluation Guide

This note explains the key arguments in `robpicker/evaluate.py` and a recommended thresholding workflow.

## Required arguments

- `--checkpoint`: Path to the trained model checkpoint (e.g., `.../checkpoint_best.pth`).
- `--data_dir`: Path to the dataset folder containing tomograms and XML files (e.g., `/path/to/my_dataset/meta` or `/path/to/my_dataset/test`).
- `--output_dir`: Directory to save predictions (and metrics if enabled).
- `--config`: Config name or module path. Use a config that matches your dataset classes and voxel spacing (e.g., `cfg_resnet34` or your custom config).

### Config
The inference tomograms should match the voxel spacing of the training tomograms and it should be set with `cfg.voxel_spacing` in the config file.

The tomogram dimension (in angstroms) should be set in the config file like this:
```
cfg.pp_x_max = 10500
cfg.pp_y_max = 10500
cfg.pp_z_max = 5500
```
The `pp_x_max`, `pp_y_max`, `pp_z_max` correspond to the X, Y, Z axis respectively.

For metric (like F-beta scores) calculation, the follow config can be set in the config file:
```
cfg.metric_beta = 1
cfg.metric_distance_multiplier = 0.5
cfg.metric_weights = {
    "ribosome80s": 1,
    "atp": 1,
}
```
The `metric_beta` denotes the beta in F-beta, so 1 means using F1 score. The `metric_distance_multiplier` means the multiplier for the `cfg.particle_radi` in config: the final distance threshold (for a prediction to be considered as true positive) is `particle_radi * metric_distance_multiplier`. And the `metric_weights` is the weight for the particle classes in calculating the overall metric.

## Common optional arguments

- `--batch_size`: Batch size for inference. Increase for faster inference if GPU memory allows.
- `--thresholds`: Comma-separated list of per-class thresholds in the order of `cfg.classes`.
- `--inference_only`: Skip metric calculation. This is useful when ground-truth annotations are not available.
- `--no_greedy_nms`: Use simple and faster non-maximum suppression.
- `--no_flip_tta`: Disable flipping the tomograms as test time augmentation. It might be useful to detect particles with certain handedness (also need to disable the flip augmentation during training).
- `--threshold_range`: Change the threshold search range if you find any of the output thresholds is out of the default range [0.1, 0.6].

## Threshold workflow

1) **Calibrate thresholds on annotated tomograms (typically the meta set).**

Run evaluation on a set with annotations (e.g., `meta/`). If you do not pass `--thresholds`, the script performs a grid search and stores the best thresholds in `metrics.json`.

```bash
robpicker-eval \
  --checkpoint /path/to/checkpoint_best.pth \
  --data_dir /path/to/my_dataset/meta \
  --output_dir ./eval_meta
```

2) **Use the calibrated thresholds for new tomograms.**

Pass the thresholds to apply them during inference. If you only have unlabeled data, add `--inference_only`.

```bash
robpicker-eval \
  --checkpoint /path/to/checkpoint_best.pth \
  --data_dir /path/to/my_dataset/test \
  --output_dir ./eval_test \
  --thresholds 0.12,0.08 \
  --inference_only
```

The script will write:
- `predictions.csv` (raw predictions)
- `predictions_thresholded.csv` (if `--thresholds` is supplied in inference-only mode)

## Notes

- Thresholds must match the order of `cfg.classes` in your config.
- If you omit `--inference_only` and annotations are present, the script computes metrics and saves `metrics.json`.
