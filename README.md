# RobPicker

RobPicker is a meta-learning 3D U-Net pipeline for cryo-ET particle picking. It leverages meta data reweighting and label correction to make supervised deep learning particle picker more robust. 
The primary training workflow uses a simple, flexible dataset layout: tomograms in MRC format plus XML files containing particle coordinates. It is very flexible because tomograms are generally available in the MRC format and annotations are available in STAR files, which can be easily converted to XML files using our `robpicker-star2xml` script (see how to use your own datasets below).

## Installation

We recommend installing RobPicker on Linux machines with Nvidia GPUs for the best experience.

RobPicker is built in Python with modules like PyTorch, you will need to setup a Python environment with dependencies. We recommend installing via [Miniforge](https://github.com/conda-forge/miniforge). This can be typically done on Linux via:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Then, create a conda environment
```
conda create --name robpicker python=3.10.19
```

Activate the conda environment
```
conda activate robpicker
```

Clone this repo to a directory and change to that directory. 
Install the package in editable mode:
```
pip install -e .
```

## Quick start
We use EMPIAR-11830 as an example to train a RobPicker model to pick ribosome and ATP synthase. First select and download the following Chlamy Visual proteomics denoised tomograms from [EMPIAR-11830](https://www.ebi.ac.uk/empiar/EMPIAR-11830/):
- train: 1287.mrc, 1317.mrc, 137.mrc, 145.mrc, 178.mrc, 1959.mrc, 2025.mrc, 2038.mrc, 2048.mrc, 2158.mrc, 2224.mrc, 2252.mrc, 269.mrc, 295.mrc
- meta: 1333.mrc, 1694.mrc
- test: 2814.mrc, 2816.mrc

Put the tomograms into the corresponding folders (train/meta/test) in the `data` folder of this repo, which contains the XML annotation files.

### Training

Change the `cfg.data_dir` in the example config file `robpicker/configs/cfg_resnet34.py` to the data path you use above. Start the training using this config file (omit the .py extension):
```
robpicker-train -C cfg_resnet34
```
The time elapse and estimated remaning time for the training are printed as `[Problem "main"] elapsed=DD HH:MM:SS eta=DD HH:MM:SS`. The training on EMPIAR-11830 can take from a few hours to 2 days, depending on the GPU capacity.

The trained models (checkpoints) are saved under the `cfg.output_dir` specified in the config: `./output/cfg_resnet34`.

### Evaluation / inference

Run inference on the meta folder containing tomograms and XMLs to determine thresholds:

```bash
robpicker-eval \
  --config cfg_resnet34 \
  --checkpoint /path/to/checkpoint_best.pth \
  --data_dir /path/to/my_dataset/meta \
  --output_dir ./output_results 
```

Run inference on the test folder containing tomograms and XMLs to calculate picking F1 score:

```bash
robpicker-eval \
  --config cfg_resnet34 \
  --checkpoint /path/to/checkpoint_best.pth \
  --data_dir /path/to/my_dataset/test \
  --output_dir ./eval_results \
  --thresholds 0.12,0.08  # use the thresholds printed above, separated by comma
```

## Using your own datasets
Follow [docs/train.md](docs/train.md) to set up training with your own tomograms and annotations. Follow [docs/evaluate.md](docs/evaluate.md) to run inference to pick particles.

## Notes

- The pipeline is CLI-first; configuration is the primary interface.
- Meta-learning modules are enabled via `robpicker/configs/meta_config.py`.


## Acknowledgement
Part of this codebase is based on the implementation of this [Kaggle solution](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/writeups/daddies-1st-place-solution-segmentation-with-partl).