# Krones Vision AI Challenge

This repository contains my work for the 1st Krones Vision AI Challenge: an
image-based approach to deciding whether an empty bottle is reusable. The
project explores semantic segmentation as an interpretable alternative to
direct image-level classification.

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
| `train_no_roi.py` | Full-image segmentation training |
| `train_no_roi_768.py` | Full-image training experiment at 768-pixel input size |
| `generate_submission.py` | Inference and submission generation for the ROI-based model |
| `generate_submission_v5.py` | Inference and submission generation for the no-ROI model |
| `diagnose_failures.py` | Validation diagnostics and misclassification analysis |
| `splits/` | Precomputed validation fold files |
| `seg/`, `seg_no_roi/`, `seg_no_roi_768/` | Experiment outputs, metrics, plots, and model checkpoints |
| `train_enriched.csv`, `category_stats.csv` | Derived training metadata and annotation statistics |

## Data and environment

The image dataset and original annotations are not included. Obtain the
authorized challenge data separately and arrange it with training images,
test images, the training annotations, and the sample submission CSV. The
training and inference scripts currently use Linux paths under
`/home/dll0202/baksh/`; update their `DATA_DIR` and `ART_DIR` constants to
match your environment before running them.

The notebooks and scripts use Python with PyTorch, segmentation-models-pytorch,
Albumentations, OpenCV, NumPy, pandas, scikit-learn, Matplotlib, and
pycocotools. GPU acceleration is optional but recommended for training.
Activate an environment with these dependencies installed before running the
commands below.

## Running the experiments

Run the EDA and split-preparation notebook first if you need to regenerate its
derived files. Then, from the repository root:

```bash
python train_no_roi.py
```

To run the 768-pixel training variant:

```bash
python train_no_roi_768.py
```

Run validation diagnostics by pointing to a trained checkpoint:

```bash
python diagnose_failures.py --ckpt seg_no_roi_768/best.pt
```

Generate a test submission with the inference script corresponding to the
checkpoint you trained. For example, `generate_submission_v5.py` expects the
no-ROI checkpoint under `seg_no_roi/best.pt`; adjust its checkpoint path if
using a different experiment. The test images and sample submission CSV must
also be available in the configured data directory.

## Data and checkpoints

Challenge data is excluded from this repository. Model checkpoint files use
Git LFS where tracked; large generated outputs may require substantial
download space. Test-time predictions and submissions should be handled
according to the challenge's data and submission rules.
