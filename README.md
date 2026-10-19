# Krones Vision AI Challenge

This repository contains my work for the 1st Krones Vision AI Challenge: an
image-based approach to deciding whether an empty bottle is reusable. The
project explores semantic segmentation as an interpretable alternative to
direct image-level classification. The Python workflows are organized as an
installable package with a single command-line entry point.

## Approach

The training pipeline learns defect and harmless-region masks from the
provided polygon annotations. It then converts predicted masks into a binary
reuse decision using area-based decision rules. This keeps the final decision
logic explicit and makes it possible to analyze which predicted regions
contributed to a result.

The experiments include a U-Net++ model with an EfficientNet-B0 encoder,
validation using predefined data splits, and comparisons between ROI-cropped
and full-image training. The full-image experiments use larger input images to
retain detail without requiring an ROI crop. Training, inference, and failure
analysis scripts and notebooks are included.

## Repository contents

| Path | Description |
| --- | --- |
| `01_eda_and_split (1).ipynb` | Exploratory analysis and preparation of training metadata and validation splits |
| `02_segmentation_training_server.ipynb` | Segmentation training notebook |
| `src/krones_vision_ai/commands/` | Training, inference, and diagnostic command implementations |
| `src/krones_vision_ai/rules.py` | Shared defect channels and decision thresholds |
| `src/krones_vision_ai/models.py` | Shared segmentation-model construction |
| `src/krones_vision_ai/paths.py` | Configurable challenge-data and artifact locations |
| `tests/` | Focused package and command-line tests |
| `pyproject.toml` | Package metadata, optional dependencies, and CLI entry point |
| `splits/` | Precomputed validation fold files |
| `seg/`, `seg_no_roi/`, `seg_no_roi_768/` | Experiment outputs, metrics, plots, and model checkpoints |
| `train_enriched.csv`, `category_stats.csv` | Derived training metadata and annotation statistics |

## Data and environment

The image dataset and original annotations are not included. Obtain the
authorized challenge data separately and arrange it with training images,
test images, the training annotations, and the sample submission CSV. By
default, the package looks for challenge data in a `dataset/` directory next
to the repository's parent directory and writes experiment artifacts to the
repository. Override these locations with `KRONES_DATA_DIR` and
`KRONES_ARTIFACT_DIR`, respectively.

Python 3.10 or newer is required. Install PyTorch for your CPU/GPU environment,
then install the package and its vision dependencies:

```bash
python -m pip install -e ".[vision,test]"
```

The vision extra includes segmentation-models-pytorch, Albumentations, OpenCV,
NumPy, pandas, scikit-learn, Matplotlib, tqdm, and pycocotools. GPU
acceleration is optional but recommended for training.

## Running workflows

Run commands from the repository root after installing the package. Set the
data paths in PowerShell as needed:

```powershell
$env:KRONES_DATA_DIR = "C:\path\to\challenge-data"
$env:KRONES_ARTIFACT_DIR = "C:\path\to\experiment-artifacts"
```

Run the EDA and split-preparation notebook first if you need to regenerate its
derived files. Train the full-image model with:

```bash
python -m krones_vision_ai train
```

Train the 768-pixel variant:

```bash
python -m krones_vision_ai train-768
```

Command-specific options are forwarded to the corresponding workflow. For
example, resume training from a checkpoint:

```bash
python -m krones_vision_ai train --resume seg_no_roi/last.pt
```

Run validation diagnostics:

```bash
python -m krones_vision_ai diagnose --ckpt seg_no_roi_768/best.pt
```

Generate submissions with the inference workflow matching the checkpoint:

```bash
python -m krones_vision_ai infer
python -m krones_vision_ai infer-roi
```

The no-ROI inference command expects `seg_no_roi/best.pt`; ROI-based inference
expects `seg/best.pt`. Both require the test images and sample submission CSV
in the configured data directory. Other available commands are `diagnose-roi`
for ROI visualization and `python -m krones_vision_ai --help` for a command
overview.

Run the focused tests with:

```bash
python -m pytest
```

## Data and checkpoints

Challenge data is excluded from this repository. Model checkpoint files use
Git LFS where tracked; large generated outputs may require substantial
download space. Test-time predictions and submissions should be handled
according to the challenge's data and submission rules.
