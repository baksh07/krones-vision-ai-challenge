"""
diagnose_failures.py — Confusion matrix and per-channel failure analysis

Loads a trained checkpoint, runs inference on the val set (fold_0 val split),
and produces:

  1. Standard 2x2 confusion matrix (TP/FP/TN/FN).
  2. Per-channel trigger statistics:
     - How often each channel fired
     - Of those firings, how many were correct (true defects) vs wrong
     - Per-channel "false alarm rate" — when channel fires, what fraction is FP
  3. Failure category breakdown:
     - Among false positives: which channel(s) were responsible?
     - Among false negatives: what was missed?
  4. A sample of the 12 most "interesting" misclassifications saved as a figure.

Outputs:
  - seg_no_roi_768/diagnostic_summary.txt   (text report)
  - seg_no_roi_768/confusion_matrix.png     (visualization)
  - seg_no_roi_768/per_channel_failures.csv (raw data)
  - seg_no_roi_768/failure_examples.png     (12 misclassified images)

Usage:
    python diagnose_failures.py --ckpt seg_no_roi_768/best.pt
    python diagnose_failures.py --ckpt seg/best.pt   # the v1 ROI model
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# Args
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, type=str,
                    help="Path to checkpoint (e.g. seg_no_roi_768/best.pt)")
parser.add_argument("--n-failure-examples", type=int, default=12,
                    help="How many misclassified examples to visualize")
parser.add_argument("--prob-threshold", type=float, default=0.5,
                    help="Probability threshold for binarizing channel masks")
args = parser.parse_args()

# ============================================================================
# Paths
# ============================================================================
DATA_DIR = Path("/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge")
ART_DIR = Path("/home/dll0202/baksh/krones")
TRAIN_IMG_DIR = DATA_DIR / "train_images"
COCO_JSON = DATA_DIR / "train_annotations.json"
FOLD_CSV = ART_DIR / "splits" / "fold_0.csv"

CKPT_PATH = Path(args.ckpt)
assert CKPT_PATH.exists(), f"Missing: {CKPT_PATH}"

# Output directory = same as where checkpoint lives
OUT_DIR = CKPT_PATH.parent
print(f"Output dir: {OUT_DIR}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

# ============================================================================
# Channel definitions (must match training)
# ============================================================================
DECISION_CHANNELS: dict[str, dict[str, Any]] = {
    "air_bubble":          {"cats": ["Air bubble"],          "thr": 500},
    "chip":                {"cats": ["Chip"],                "thr": 200},
    "contamination_light": {"cats": ["Contamination light"], "thr": 180},
    "glass_imperfection":  {"cats": ["Glass imperfection"],  "thr": 100},
    "scuffing":            {"cats": ["Scuffing"],            "thr": 75_000},
    "scuffing_heavy":      {"cats": ["Scuffing heavy"],      "thr": 1_200},
    "always_faulty": {
        "cats": [
            "Break / Crack", "Circlip", "Contamination dark", "Crown cap",
            "Foil / Semitransparent",
            "Foreign object - manual cleaning", "Foreign object - washing machine",
            "Glass shard", "Insect", "Label", "Liquid", "Mold",
            "No base visible", "Paint residue", "Straw", "Yeast residue",
        ],
        "thr": 1,
    },
    "harmless": {
        "cats": ["Embossing", "Foam residue", "Water drop"],
        "thr": None,
    },
}
CHANNEL_NAMES = list(DECISION_CHANNELS.keys())
CHANNEL_THRESHOLDS = [DECISION_CHANNELS[n]["thr"] for n in CHANNEL_NAMES]
N_CHANNELS = len(CHANNEL_NAMES)
AGGREGATED_CHANNEL_INDICES = {
    CHANNEL_NAMES.index("scuffing"),
    CHANNEL_NAMES.index("scuffing_heavy"),
}


# ============================================================================
# Load checkpoint
# ============================================================================
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
saved_cfg = ckpt["cfg"]
print(f"  Saved val F1       : {ckpt['val_f1']:.4f}")
print(f"  Saved at epoch     : {ckpt['epoch']}")
print(f"  img_size           : {saved_cfg['img_size']}")
print(f"  use_roi_crop       : {saved_cfg.get('use_roi_crop', '?')}")
print(f"  encoder            : {saved_cfg['encoder_name']}")
print(f"  arch               : {saved_cfg['arch']}\n")


@dataclass
class InferCfg:
    img_size: int = saved_cfg["img_size"]
    encoder_name: str = saved_cfg["encoder_name"]
    encoder_weights: str = saved_cfg["encoder_weights"]
    arch: str = saved_cfg["arch"]
    use_roi_crop: bool = saved_cfg.get("use_roi_crop", False)
    orig_h: int = saved_cfg.get("orig_h", 1024)
    orig_w: int = saved_cfg.get("orig_w", 1280)
    aggregate_scuffing: bool = bool(ckpt.get("aggregate_scuffing", False))
    batch_size: int = 8

    @property
    def area_scale(self) -> float:
        return (self.orig_h * self.orig_w) / (self.img_size * self.img_size)


CFG = InferCfg()


# ============================================================================
# Build model and load weights
# ============================================================================
def build_model(cfg, n_classes):
    arch_map = {"unetplusplus": smp.UnetPlusPlus, "unet": smp.Unet, "fpn": smp.FPN}
    Arch = arch_map[cfg.arch]
    return Arch(encoder_name=cfg.encoder_name, encoder_weights=None,
                in_channels=3, classes=n_classes)


model = build_model(CFG, n_classes=N_CHANNELS).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Model loaded ✓\n")


# ============================================================================
# COCO indexing for ROI lookup (only used if use_roi_crop)
# ============================================================================
print("Loading COCO annotations for ROI lookup...")
with open(COCO_JSON) as f:
    coco = json.load(f)


def _normalise(name: str) -> str:
    return name.replace("\u2013", "-").replace("\u2014", "-").strip()


cat_id_to_name = {c["id"]: _normalise(c["name"]) for c in coco["categories"]}
img_id_to_meta = {im["id"]: im for im in coco["images"]}
ROI_ALIASES = {"roi", "region of interest"}
roi_cat_ids = {cid for cid, n in cat_id_to_name.items() if n.lower() in ROI_ALIASES}
roi_by_image = {}
for ann in coco["annotations"]:
    if ann["category_id"] in roi_cat_ids:
        meta = img_id_to_meta.get(ann["image_id"])
        if meta:
            x, y, w, h = ann["bbox"]
            roi_by_image[meta["file_name"]] = (int(x), int(y), int(w), int(h))
print(f"Loaded {len(roi_by_image):,} ROI bboxes\n")


# ============================================================================
# Val dataset (mirror what training did)
# ============================================================================
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_val_aug(size: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class ValDataset(Dataset):
    def __init__(self, df, img_dir, transform, use_roi, roi_by_image):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.use_roi = use_roi
        self.roi_by_image = roi_by_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["image_id"]
        img = cv2.imread(str(self.img_dir / fname), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"{fname}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.use_roi:
            roi = self.roi_by_image.get(fname)
            if roi is not None:
                x, y, w, h = roi
                H, W = img.shape[:2]
                pad = 10
                x0 = max(0, x - pad); y0 = max(0, y - pad)
                x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
                if x1 > x0 and y1 > y0:
                    img = img[y0:y1, x0:x1]

        out = self.transform(image=img)
        return {"image": out["image"], "image_id": fname,
                "target": int(row["target"])}


fold = pd.read_csv(FOLD_CSV)
val_df = fold[fold["split"] == "val"].reset_index(drop=True)
print(f"Val set size: {len(val_df):,}")
print(f"  target=0 (Reusable):     {(val_df['target']==0).sum():,}  ({(val_df['target']==0).mean()*100:.1f}%)")
print(f"  target=1 (Not Reusable): {(val_df['target']==1).sum():,}  ({(val_df['target']==1).mean()*100:.1f}%)")
print()

val_ds = ValDataset(val_df, TRAIN_IMG_DIR, build_val_aug(CFG.img_size),
                    CFG.use_roi_crop, roi_by_image)
val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)


# ============================================================================
# Decision function (returns verdict + per-channel triggers for diagnosis)
# ============================================================================
@torch.no_grad()
def predict_with_reasons(logits, channel_thresholds, prob_threshold,
                          scale_to_full, aggregated_channels):
    """For each image, return verdict + a list of (channel_idx, blob_area) triggers."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    bin_masks = (probs > prob_threshold).astype(np.uint8)
    B, C, H, W = bin_masks.shape
    out_verdicts = np.zeros(B, dtype=np.uint8)
    out_triggers = [[] for _ in range(B)]
    for b in range(B):
        for c in range(C):
            thr = channel_thresholds[c]
            if thr is None:
                continue
            mask = bin_masks[b, c]
            if not mask.any():
                continue
            n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n_lbl <= 1:
                continue
            blob_areas = stats[1:, cv2.CC_STAT_AREA]
            if c in aggregated_channels:
                area_used = int(blob_areas.sum())
            else:
                area_used = int(blob_areas.max())
            scaled = area_used * scale_to_full
            if scaled > thr:
                out_verdicts[b] = 1
                out_triggers[b].append((c, int(scaled)))
    return out_verdicts, out_triggers, probs


# ============================================================================
# Run inference on the entire val set
# ============================================================================
print("Running inference on val set...")
all_y_true, all_y_pred, all_ids, all_triggers = [], [], [], []
agg = AGGREGATED_CHANNEL_INDICES if CFG.aggregate_scuffing else set()

with torch.no_grad():
    with torch.amp.autocast("cuda", enabled=True):
        for batch in tqdm(val_loader, desc="val"):
            img = batch["image"].to(DEVICE, non_blocking=True)
            logits = model(img)
            verdicts, triggers, _ = predict_with_reasons(
                logits, CHANNEL_THRESHOLDS, args.prob_threshold,
                CFG.area_scale, agg
            )
            all_y_true.extend(batch["target"].tolist())
            all_y_pred.extend(verdicts.tolist())
            all_ids.extend(batch["image_id"])
            all_triggers.extend(triggers)

y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)
print(f"\nDone. {len(y_true):,} val images processed.\n")


# ============================================================================
# Confusion matrix and overall metrics
# ============================================================================
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
total = tn + fp + fn + tp
acc = (tn + tp) / total
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("=" * 60)
print("OVERALL METRICS")
print("=" * 60)
print(f"  TP (correctly flagged defective)     : {tp:>5,}  ({100*tp/total:.1f}%)")
print(f"  TN (correctly flagged reusable)      : {tn:>5,}  ({100*tn/total:.1f}%)")
print(f"  FP (false alarm — said defect, isn't): {fp:>5,}  ({100*fp/total:.1f}%)  ← over-firing")
print(f"  FN (missed defect — said safe, isn't): {fn:>5,}  ({100*fn/total:.1f}%)  ← critical errors")
print(f"  --")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1       : {f1:.4f}")
print()


# ============================================================================
# Per-channel firing analysis
# ============================================================================
# For each channel: how often did it fire? When it fired, was the verdict right?
print("=" * 60)
print("PER-CHANNEL FIRING ANALYSIS")
print("=" * 60)

channel_stats = defaultdict(lambda: {"n_fired": 0, "fired_on_true_defect": 0, "fired_on_actual_clean": 0})

for i in range(len(y_true)):
    triggers = all_triggers[i]
    actual = y_true[i]
    fired_channels = set(c for c, _ in triggers)
    for c in fired_channels:
        channel_stats[c]["n_fired"] += 1
        if actual == 1:
            channel_stats[c]["fired_on_true_defect"] += 1
        else:
            channel_stats[c]["fired_on_actual_clean"] += 1

per_channel_data = []
print(f"{'Channel':<22} {'n_fired':>8} {'on_defect':>10} {'on_clean':>10} {'precision':>10}")
print("-" * 64)
for c in range(N_CHANNELS):
    if CHANNEL_THRESHOLDS[c] is None:
        continue   # harmless never decides
    s = channel_stats[c]
    n = s["n_fired"]
    p = s["fired_on_true_defect"] / n if n > 0 else 0.0
    print(f"{CHANNEL_NAMES[c]:<22} {n:>8} {s['fired_on_true_defect']:>10} {s['fired_on_actual_clean']:>10} {p:>10.4f}")
    per_channel_data.append({
        "channel": CHANNEL_NAMES[c],
        "threshold_px": CHANNEL_THRESHOLDS[c],
        "n_fired": n,
        "fired_on_true_defect": s["fired_on_true_defect"],
        "fired_on_actual_clean": s["fired_on_actual_clean"],
        "precision": round(p, 4),
    })
print()


# ============================================================================
# Failure analysis: what triggered the FPs and what was missed in FNs
# ============================================================================
print("=" * 60)
print("FALSE POSITIVE BREAKDOWN (model said 1, truth was 0)")
print("=" * 60)
fp_triggers_by_channel = Counter()
for i in range(len(y_true)):
    if y_true[i] == 0 and y_pred[i] == 1:
        for c, area in all_triggers[i]:
            fp_triggers_by_channel[CHANNEL_NAMES[c]] += 1

if fp_triggers_by_channel:
    total_fp_triggers = sum(fp_triggers_by_channel.values())
    print(f"{'Channel':<22} {'count':>8} {'%_of_FPs':>10}")
    print("-" * 44)
    for ch, count in fp_triggers_by_channel.most_common():
        pct = 100.0 * count / total_fp_triggers
        print(f"{ch:<22} {count:>8} {pct:>9.1f}%")
else:
    print("No false positives.")
print()

print("=" * 60)
print("FALSE NEGATIVE COUNT (model said 0, truth was 1)")
print("=" * 60)
print(f"  Number of FNs: {fn}")
print(f"  These are critical errors — defective bottles passed through.")
print(f"  We can't easily diagnose without ground-truth masks (would need to")
print(f"  rasterize the COCO annotations and compare to model output).")
print()


# ============================================================================
# Save text report and CSV
# ============================================================================
report_path = OUT_DIR / "diagnostic_summary.txt"
with open(report_path, "w") as f:
    f.write(f"Checkpoint: {CKPT_PATH}\n")
    f.write(f"Val F1: {f1:.4f}\n")
    f.write(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}\n\n")
    f.write("Per-channel firing:\n")
    for d in per_channel_data:
        f.write(f"  {d}\n")
    f.write("\nFP triggers by channel:\n")
    for ch, count in fp_triggers_by_channel.most_common():
        f.write(f"  {ch}: {count}\n")
print(f"✓ Saved {report_path}")

per_channel_csv = OUT_DIR / "per_channel_failures.csv"
pd.DataFrame(per_channel_data).to_csv(per_channel_csv, index=False)
print(f"✓ Saved {per_channel_csv}")


# ============================================================================
# Confusion matrix visualization
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
cm_normalized = cm.astype(float) / cm.sum()
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted: Reusable", "Predicted: Not Reusable"])
ax.set_yticklabels(["True: Reusable", "True: Not Reusable"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (val set, F1={f1:.4f})")
for i in range(2):
    for j in range(2):
        cell = cm[i, j]
        pct = cm_normalized[i, j] * 100
        color = "white" if cell > cm.max() / 2 else "black"
        ax.text(j, i, f"{cell:,}\n({pct:.1f}%)",
                ha="center", va="center", color=color, fontsize=14, fontweight="bold")
plt.colorbar(im)
plt.tight_layout()
cm_path = OUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=120, bbox_inches="tight")
print(f"✓ Saved {cm_path}")
plt.close()


# ============================================================================
# Per-channel firing visualization
# ============================================================================
if per_channel_data:
    fig, ax = plt.subplots(figsize=(12, 6))
    channels = [d["channel"] for d in per_channel_data]
    fired_def = [d["fired_on_true_defect"] for d in per_channel_data]
    fired_clean = [d["fired_on_actual_clean"] for d in per_channel_data]
    x = np.arange(len(channels))
    width = 0.35
    ax.bar(x - width/2, fired_def, width, label="Fired on true defect (good)", color="#2ca02c")
    ax.bar(x + width/2, fired_clean, width, label="Fired on clean image (false alarm)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=30, ha="right")
    ax.set_ylabel("Number of images channel fired on")
    ax.set_title("Per-channel firing — true defects vs false alarms")
    ax.legend()
    plt.tight_layout()
    ch_path = OUT_DIR / "per_channel_firings.png"
    plt.savefig(ch_path, dpi=120, bbox_inches="tight")
    print(f"✓ Saved {ch_path}")
    plt.close()


# ============================================================================
# Failure examples — show 12 misclassified images
# ============================================================================
print(f"\nGenerating failure examples ({args.n_failure_examples} images)...")
fp_indices = [i for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1]
fn_indices = [i for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0]
n_each = args.n_failure_examples // 2
sample_fp = fp_indices[:n_each]
sample_fn = fn_indices[:n_each]
samples = sample_fp + sample_fn

if samples:
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if rows > 1 else axes
    for plot_i, val_i in enumerate(samples):
        fname = all_ids[val_i]
        img = cv2.imread(str(TRAIN_IMG_DIR / fname), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kind = "FP" if y_true[val_i] == 0 else "FN"
        triggers_str = ", ".join(f"{CHANNEL_NAMES[c]}({a})" for c, a in all_triggers[val_i])
        if not triggers_str:
            triggers_str = "(none)"
        ax = axes[plot_i] if isinstance(axes, np.ndarray) else axes
        ax.imshow(img)
        ax.set_title(f"{kind}: true={y_true[val_i]}, pred={y_pred[val_i]}\n"
                      f"{fname[:30]}\nTriggers: {triggers_str[:40]}",
                      fontsize=8)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    plt.tight_layout()
    fail_path = OUT_DIR / "failure_examples.png"
    plt.savefig(fail_path, dpi=100, bbox_inches="tight")
    print(f"✓ Saved {fail_path}")
    plt.close()

print("\n=== DIAGNOSTIC COMPLETE ===")
print(f"All outputs in: {OUT_DIR}")