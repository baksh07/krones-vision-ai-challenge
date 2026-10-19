"""
train_no_roi.py — Krones Vision AI Challenge

Production-grade training script for tmux execution.

Key design decisions:
- use_roi_crop = False           → eliminates the train-test ROI mismatch
- img_size = 768                  → bigger input compensates for not cropping the bottle
- Saves best.pt AND last.pt every epoch (not just at end)
- Writes history.csv after every epoch
- Supports --resume from any saved checkpoint
- persistent_workers=True         → eliminates the worker-shutdown crashes

Usage:
    # Fresh start
    python -m krones_vision_ai train-768

    # Resume from a checkpoint
    python -m krones_vision_ai train-768 --resume seg_no_roi_768/last.pt

    # In tmux (recommended)
    tmux new -s train
    source ~/baksh/krones/venv/bin/activate
    cd ~/baksh/krones
    python -m krones_vision_ai train-768 2>&1 | tee training.log
    # detach: Ctrl+B then D
    # reattach: tmux attach -t train
"""

from __future__ import annotations


def main() -> None:
    import argparse

    # ============================================================================
    # Args
    # ============================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt to resume from (preserves optimizer + scheduler state)",
    )
    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
        help="Path to checkpoint .pt to fine-tune from (loads model weights ONLY; "
        "fresh optimizer and scheduler with the new LR/epochs)",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--img-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop training if val F1 hasn't improved for this many epochs",
    )
    args = parser.parse_args()

    import json
    import time
    from collections import defaultdict
    from dataclasses import asdict, dataclass
    from pathlib import Path
    from typing import Any

    import albumentations as A
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from albumentations.pytorch import ToTensorV2
    from sklearn.metrics import f1_score, precision_score, recall_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    from krones_vision_ai.paths import ARTIFACT_ROOT, DATA_ROOT

    # ============================================================================
    # Paths
    # ============================================================================
    DATA_DIR = DATA_ROOT
    ART_DIR = ARTIFACT_ROOT
    OUT_DIR = (
        ART_DIR / "seg_no_roi_768"
    )  # NEW directory; preserves seg_no_roi from the 512² run
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_IMG_DIR = DATA_DIR / "train_images"
    COCO_JSON = DATA_DIR / "train_annotations.json"
    FOLD_CSV = ART_DIR / "splits" / "fold_0.csv"

    from krones_vision_ai.paths import require_paths

    require_paths(
        {
            "training images": TRAIN_IMG_DIR,
            "COCO annotations": COCO_JSON,
            "validation fold": FOLD_CSV,
        }
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}, "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # ============================================================================
    # Config
    # ============================================================================
    @dataclass
    class Cfg:
        img_size: int = args.img_size
        batch_size: int = args.batch_size
        num_workers: int = 4  # 4 instead of 8 for stability
        epochs: int = args.epochs
        lr: float = args.lr
        weight_decay: float = 1e-5
        encoder_name: str = "efficientnet-b0"
        encoder_weights: str = "imagenet"
        arch: str = "unetplusplus"
        amp: bool = True

        # Critical change vs v1: do NOT crop to ROI during training.
        # The model will learn to handle the full scene natively, eliminating
        # the train-test ROI-crop mismatch that gave us a 0.10 F1 gap on Kaggle.
        use_roi_crop: bool = False

        aggregate_scuffing: bool = False  # default rule-faithful; ablation later

        orig_h: int = 1024
        orig_w: int = 1280

        @property
        def area_scale(self) -> float:
            return (self.orig_h * self.orig_w) / (self.img_size * self.img_size)

    CFG = Cfg()
    print("\nConfig:")
    print(f"  img_size       : {CFG.img_size}")
    print(f"  batch_size     : {CFG.batch_size}")
    print(f"  num_workers    : {CFG.num_workers}")
    print(f"  epochs         : {CFG.epochs}")
    print(f"  lr             : {CFG.lr}")
    print(f"  use_roi_crop   : {CFG.use_roi_crop}  (key change from v1)")
    print(f"  area_scale     : {CFG.area_scale:.4f}")
    print(f"  output dir     : {OUT_DIR}")

    # ============================================================================
    # Channels (same scheme as v1 — collapse 26 categories into 8 decision classes)
    # ============================================================================
    from krones_vision_ai.rules import (
        AGGREGATED_CHANNEL_INDICES,
        CHANNEL_NAMES,
        CHANNEL_THRESHOLDS,
        DECISION_CHANNELS,
        N_CHANNELS,
    )

    def _normalise(name: str) -> str:
        return name.replace("\u2013", "-").replace("\u2014", "-").strip()

    # ============================================================================
    # COCO indexing
    # ============================================================================
    print("\nLoading COCO annotations...")
    with open(COCO_JSON) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: _normalise(c["name"]) for c in coco["categories"]}
    img_id_to_meta = {im["id"]: im for im in coco["images"]}

    ROI_ALIASES = {"roi", "region of interest"}
    roi_cat_ids = {cid for cid, n in cat_id_to_name.items() if n.lower() in ROI_ALIASES}

    # Channel mapping
    name_to_channel: dict[str, int] = {}
    for ch_idx, name in enumerate(CHANNEL_NAMES):
        for cat_name in DECISION_CHANNELS[name]["cats"]:
            name_to_channel[_normalise(cat_name)] = ch_idx

    print(
        f"Loaded {len(coco['annotations']):,} annotations across {len(coco['images']):,} images"
    )
    print(f"ROI category ids: {roi_cat_ids}")

    polys_by_image: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        img_meta = img_id_to_meta.get(ann["image_id"])
        if img_meta is None:
            continue
        fname = img_meta["file_name"]
        cat_name = cat_id_to_name.get(ann["category_id"], "")
        if ann["category_id"] in roi_cat_ids:
            continue
        ch_idx = name_to_channel.get(cat_name)
        if ch_idx is None:
            continue
        polys_by_image[fname].append((ch_idx, ann.get("segmentation", [])))

    print(f"Built polygons for {len(polys_by_image):,} images")

    # ============================================================================
    # Mask rasterization (fixed for OpenCV 4.13 contiguous-array requirement)
    # ============================================================================
    def rasterize_polygons(polys, height, width, n_channels):
        mask_chw = np.zeros((n_channels, height, width), dtype=np.uint8)
        for ch_idx, seg in polys:
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    pts = (
                        np.asarray(poly, dtype=np.float32)
                        .reshape(-1, 2)
                        .round()
                        .astype(np.int32)
                    )
                    cv2.fillPoly(mask_chw[ch_idx], [pts], 1)
            elif isinstance(seg, dict):
                from pycocotools import mask as mask_utils

                rle = seg
                if isinstance(rle.get("counts"), list):
                    rle = mask_utils.frPyObjects([rle], height, width)[0]
                decoded_mask = mask_utils.decode(rle)
                mask_chw[ch_idx] |= decoded_mask.astype(np.uint8)
        return np.ascontiguousarray(mask_chw.transpose(1, 2, 0))

    # ============================================================================
    # Augmentations
    # ============================================================================
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def build_train_aug(size: int) -> A.Compose:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(
                    min_height=size,
                    min_width=size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=24,
                    max_width=24,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.3,
                ),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )

    def build_val_aug(size: int) -> A.Compose:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(
                    min_height=size,
                    min_width=size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )

    # ============================================================================
    # Dataset (no ROI crop — that's the whole point of this script)
    # ============================================================================
    class KronesSegDataset(Dataset):
        def __init__(self, df, img_dir, polys_by_image, n_channels, transform):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.polys_by_image = polys_by_image
            self.n_channels = n_channels
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            fname = row["image_id"]
            img = cv2.imread(str(self.img_dir / fname), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read {self.img_dir / fname}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            polys = self.polys_by_image.get(fname, [])
            mask = rasterize_polygons(polys, H, W, self.n_channels)

            # NO ROI CROP — full image goes to transform

            out = self.transform(image=img, mask=mask)
            mask_tensor = out["mask"].permute(2, 0, 1).float()
            return {
                "image": out["image"],
                "mask": mask_tensor,
                "target": torch.tensor(int(row["target"]), dtype=torch.long),
                "image_id": fname,
            }

    # ============================================================================
    # Model
    # ============================================================================
    from krones_vision_ai.models import build_model

    # ============================================================================
    # Loss — weighted Dice + BCE
    # ============================================================================
    CHANNEL_LOSS_WEIGHTS = torch.tensor(
        [
            2.0,  # air_bubble
            2.0,  # chip
            2.0,  # contamination_light
            3.0,  # glass_imperfection
            1.0,  # scuffing
            1.5,  # scuffing_heavy
            1.0,  # always_faulty
            0.5,  # harmless
        ]
    )

    class WeightedDiceBCE(nn.Module):
        def __init__(self, channel_weights):
            super().__init__()
            self.register_buffer("cw", channel_weights.clone().float())

        def forward(self, logits, mask):
            probs = torch.sigmoid(logits)
            # Per-channel Dice
            eps = 1e-5
            dims = (0, 2, 3)
            intersection = (probs * mask).sum(dim=dims)
            denom = (probs + mask).sum(dim=dims) + eps
            dice = 1 - (2 * intersection + eps) / denom  # (C,)
            dice_loss = (dice * self.cw.to(probs.device)).mean()
            # Per-channel BCE
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, mask, reduction="none"
            )
            bce = (bce.mean(dim=(0, 2, 3)) * self.cw.to(probs.device)).mean()
            return 0.5 * dice_loss + 0.5 * bce

    # ============================================================================
    # Decision function
    # ============================================================================
    from krones_vision_ai.decisions import predict_from_logits

    # ============================================================================
    # DataLoaders
    # ============================================================================
    def make_loaders(cfg):
        fold = pd.read_csv(FOLD_CSV)
        train_df = fold[fold["split"] == "train"].reset_index(drop=True)
        val_df = fold[fold["split"] == "val"].reset_index(drop=True)
        print(f"\nFold 0 split: {len(train_df):,} train + {len(val_df):,} val")

        train_ds = KronesSegDataset(
            train_df,
            TRAIN_IMG_DIR,
            polys_by_image,
            N_CHANNELS,
            build_train_aug(cfg.img_size),
        )
        val_ds = KronesSegDataset(
            val_df,
            TRAIN_IMG_DIR,
            polys_by_image,
            N_CHANNELS,
            build_val_aug(cfg.img_size),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,  # KEY: don't tear down workers between epochs
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return train_loader, val_loader

    # ============================================================================
    # Evaluate
    # ============================================================================
    @torch.no_grad()
    def evaluate(model, loader, cfg):
        model.eval()
        y_true_all, y_pred_all = [], []
        agg = AGGREGATED_CHANNEL_INDICES if cfg.aggregate_scuffing else set()
        for batch in tqdm(loader, desc="val", leave=False):
            img = batch["image"].to(DEVICE, non_blocking=True)
            y_true = batch["target"].numpy()
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                logits = model(img)
            verdicts = predict_from_logits(
                logits,
                CHANNEL_THRESHOLDS,
                area_scale=cfg.area_scale,
                aggregated_channels=agg,
            ).predictions
            y_true_all.append(y_true)
            y_pred_all.append(verdicts)
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        return {
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
        }

    # ============================================================================
    # Training
    # ============================================================================
    def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, val_f1, cfg):
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
            if scheduler is not None
            else None,
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "val_f1": val_f1,
            "cfg": asdict(cfg),
            "channel_names": CHANNEL_NAMES,
            "channel_thresholds": CHANNEL_THRESHOLDS,
            "area_scale": cfg.area_scale,
            "aggregate_scuffing": cfg.aggregate_scuffing,
        }
        torch.save(ckpt, path)

    def append_history(path: Path, row: dict):
        """Append one row to the history CSV — write after every epoch, not at the end."""
        df = pd.DataFrame([row])
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, mode="w", header=True, index=False)

    def train(cfg, resume_path=None, finetune_path=None):
        train_loader, val_loader = make_loaders(cfg)

        model = build_model(cfg, n_classes=N_CHANNELS).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=cfg.epochs * len(train_loader),
            pct_start=0.05,
            anneal_strategy="cos",
        )
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
        loss_fn = WeightedDiceBCE(CHANNEL_LOSS_WEIGHTS).to(DEVICE)

        start_epoch = 1
        best_f1 = 0.0
        no_improve = 0  # for early stopping

        # Mode 1: full resume (preserves optimizer + scheduler state)
        if resume_path is not None:
            print(f"\n>>> Resuming from {resume_path} (full state)")
            ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_f1 = ckpt.get("val_f1", 0.0)
            print(
                f"    Resumed at epoch {start_epoch}, prior best F1 = {best_f1:.4f}\n"
            )

        # Mode 2: fine-tune (model weights ONLY; fresh optimizer + scheduler with new LR)
        elif finetune_path is not None:
            print(f"\n>>> Fine-tuning from {finetune_path}")
            print(
                f"    Loading model weights only — fresh optimizer + scheduler at lr={cfg.lr}"
            )
            ckpt = torch.load(finetune_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            prior_f1 = ckpt.get("val_f1", 0.0)
            prior_epoch = ckpt.get("epoch", "?")
            best_f1 = prior_f1  # so we don't save worse-than-prior as new "best"
            print(
                f"    Loaded weights from epoch {prior_epoch}, prior val F1 = {prior_f1:.4f}"
            )
            print(f"    Will only save best.pt if F1 exceeds {prior_f1:.4f}\n")

        history_path = OUT_DIR / "history.csv"
        best_path = OUT_DIR / "best.pt"
        last_path = OUT_DIR / "last.pt"

        for epoch in range(start_epoch, cfg.epochs + 1):
            # --- Train one epoch ---
            model.train()
            t0 = time.time()
            running_loss = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=True)
            for batch in pbar:
                img = batch["image"].to(DEVICE, non_blocking=True)
                mask = batch["mask"].to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=cfg.amp):
                    logits = model(img)
                    loss = loss_fn(logits, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(
                    loss=f"{running_loss / n_batches:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            train_loss = running_loss / max(1, n_batches)
            epoch_time = time.time() - t0

            # --- Validate ---
            metrics = evaluate(model, val_loader, cfg)
            print(
                f"  → val F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                f"(epoch took {epoch_time:.1f}s)"
            )

            # --- Save checkpoints (always last.pt, also best.pt if improved) ---
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics["f1"],
                cfg,
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                no_improve = 0
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    metrics["f1"],
                    cfg,
                )
                print(f"  ✅ saved {best_path} (best F1={best_f1:.4f})")
            else:
                no_improve += 1
                print(
                    f"  ⚪ no improvement ({no_improve}/{args.patience} until early stop)"
                )
            print(f"  💾 saved {last_path} (last)")

            # --- Append to history (after every epoch — not just at end) ---
            append_history(
                history_path,
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_f1": metrics["f1"],
                    "val_precision": metrics["precision"],
                    "val_recall": metrics["recall"],
                    "epoch_time_sec": round(epoch_time, 1),
                },
            )

            # --- Early stopping check ---
            if no_improve >= args.patience:
                print(f"\n🛑 Early stop: no improvement for {no_improve} epochs.")
                print(f"   Best F1 was {best_f1:.4f} at epoch {epoch - no_improve}.")
                break

        print("\n=== TRAINING COMPLETE ===")
        print(f"Best val F1: {best_f1:.4f}")
        print(f"Checkpoints in: {OUT_DIR}")

    train(CFG, resume_path=args.resume, finetune_path=args.finetune_from)


if __name__ == "__main__":
    main()
