"""
generate_submission_v5.py — for the no-ROI-crop model

Loads ~/baksh/krones/seg_no_roi/best.pt (the model trained without ROI cropping),
runs inference on all 4,418 test images, writes submission.csv.

This script is much simpler than v4 because train and test now use the same
preprocessing pipeline: just resize the full 1280x1024 image to img_size (768
by default), no ROI detection, no special-casing.

Usage:
    cd ~/baksh/krones
    source venv/bin/activate
    python -m krones_vision_ai infer
"""
from __future__ import annotations

def main() -> None:
    from krones_vision_ai.paths import ARTIFACT_ROOT, DATA_ROOT

    from dataclasses import dataclass
    from pathlib import Path

    import albumentations as A
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    # ============================================================================
    # Paths
    # ============================================================================
    DATA_DIR = DATA_ROOT
    ART_DIR = ARTIFACT_ROOT
    OUT_DIR = ART_DIR / "seg_no_roi"
    CKPT_PATH = OUT_DIR / "best.pt"

    TEST_IMG_DIR = DATA_DIR / "test_images"
    SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

    assert CKPT_PATH.exists(), f"Missing checkpoint: {CKPT_PATH}"
    assert TEST_IMG_DIR.exists(), f"Missing test_images: {TEST_IMG_DIR}"
    assert SAMPLE_SUB.exists(), f"Missing sample_submission.csv: {SAMPLE_SUB}"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")


    # ============================================================================
    # Channel definitions (must match training)
    # ============================================================================
    from krones_vision_ai.rules import (
        AGGREGATED_CHANNEL_INDICES,
        CHANNEL_NAMES,
        CHANNEL_THRESHOLDS,
        N_CHANNELS,
    )


    # ============================================================================
    # Load checkpoint
    # ============================================================================
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    saved_cfg = ckpt["cfg"]
    print(f"  Saved val F1       : {ckpt['val_f1']:.4f}")
    print(f"  Saved at epoch     : {ckpt['epoch']}")
    print(f"  aggregate_scuffing : {ckpt.get('aggregate_scuffing', False)}")
    print(f"  img_size           : {saved_cfg['img_size']}")
    print(f"  encoder            : {saved_cfg['encoder_name']}")
    print(f"  arch               : {saved_cfg['arch']}")
    print(f"  use_roi_crop       : {saved_cfg.get('use_roi_crop', '?')} (should be False)")


    @dataclass
    class InferCfg:
        img_size: int = saved_cfg["img_size"]
        encoder_name: str = saved_cfg["encoder_name"]
        encoder_weights: str = saved_cfg["encoder_weights"]
        arch: str = saved_cfg["arch"]
        orig_h: int = saved_cfg.get("orig_h", 1024)
        orig_w: int = saved_cfg.get("orig_w", 1280)
        batch_size: int = 8   # conservative for 768²
        num_workers: int = 4
        aggregate_scuffing: bool = bool(ckpt.get("aggregate_scuffing", False))

        @property
        def area_scale(self) -> float:
            return (self.orig_h * self.orig_w) / (self.img_size * self.img_size)


    CFG = InferCfg()
    print(f"  area_scale (derived): {CFG.area_scale:.4f}")
    print(f"  inference batch    : {CFG.batch_size}")


    # ============================================================================
    # Build model and load weights
    # ============================================================================
    from krones_vision_ai.models import build_model


    model = build_model(CFG, n_classes=N_CHANNELS, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("  Model loaded ✓\n")


    # ============================================================================
    # Test dataset — JUST RESIZE, no ROI logic
    # ============================================================================
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)


    def build_test_aug(size: int) -> A.Compose:
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])


    class TestDataset(Dataset):
        """Trivial: just read image, resize, normalize, return.
        Same preprocessing as training, by design."""
        def __init__(self, df, img_dir, transform):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
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
            out = self.transform(image=img)
            return {"image": out["image"], "image_id": fname}


    sample_sub = pd.read_csv(SAMPLE_SUB)
    print(f"Sample submission has {len(sample_sub):,} rows\n")

    test_ds = TestDataset(sample_sub[["image_id"]], TEST_IMG_DIR, build_test_aug(CFG.img_size))
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )


    # ============================================================================
    # Decision function
    # ============================================================================
    @torch.no_grad()
    def predict_binary(logits, channel_thresholds, prob_threshold=0.5,
                        scale_to_full=1.0, aggregated_channels=None):
        if aggregated_channels is None:
            aggregated_channels = set()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        bin_masks = (probs > prob_threshold).astype(np.uint8)
        B, C, H, W = bin_masks.shape
        out = np.zeros(B, dtype=np.uint8)
        max_probs = probs.reshape(B, C, -1).max(axis=2)
        for b in range(B):
            verdict = 0
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
                    verdict = 1
                    break
            out[b] = verdict
        return out, max_probs


    # ============================================================================
    # Run inference (with torch.no_grad to prevent OOM — same as v4 fix)
    # ============================================================================
    print("=" * 60)
    print(f"Running inference on {len(test_ds):,} test images")
    print(f"  aggregate_scuffing : {CFG.aggregate_scuffing}")
    print(f"  img_size           : {CFG.img_size}")
    print(f"  batch_size         : {CFG.batch_size}")
    print("=" * 60)

    agg = AGGREGATED_CHANNEL_INDICES if CFG.aggregate_scuffing else set()
    all_ids, all_preds, all_probs = [], [], []

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            for batch in tqdm(test_loader, desc="inference", total=len(test_loader)):
                img = batch["image"].to(DEVICE, non_blocking=True)
                ids = batch["image_id"]
                logits = model(img)
                verdicts, max_probs = predict_binary(
                    logits, CHANNEL_THRESHOLDS, prob_threshold=0.5,
                    scale_to_full=CFG.area_scale, aggregated_channels=agg,
                )
                all_ids.extend(ids)
                all_preds.extend(verdicts.tolist())
                all_probs.append(max_probs)
                del logits, img

    torch.cuda.empty_cache()
    print(f"\n✓ Inference complete on {len(all_ids):,} images")


    # ============================================================================
    # Build and save submission
    # ============================================================================
    sub = pd.DataFrame({"image_id": all_ids, "target": all_preds})
    sample_set = set(sample_sub["image_id"])
    ours_set = set(sub["image_id"])
    assert sample_set == ours_set, "Image-id mismatch!"
    sub = sub.set_index("image_id").reindex(sample_sub["image_id"]).reset_index()

    n_pos = int(sub["target"].sum())
    n_neg = len(sub) - n_pos
    print(f"\nPrediction distribution:")
    print(f"  Reusable (0)    : {n_neg:>5,}  ({100*n_neg/len(sub):.1f}%)")
    print(f"  Not Reusable (1): {n_pos:>5,}  ({100*n_pos/len(sub):.1f}%)")
    print(f"  Total           : {len(sub):>5,}")

    ratio = n_pos / len(sub)
    if ratio < 0.30 or ratio > 0.85:
        print(f"\n⚠️  WARNING: ratio {ratio:.1%} differs sharply from train (58.3%).")
        print("   Worth investigating before submitting.")

    sub_path = OUT_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n✓ Saved submission: {sub_path}")

    probs_arr = np.concatenate(all_probs, axis=0)
    probs_df = pd.DataFrame(probs_arr, columns=CHANNEL_NAMES)
    probs_df.insert(0, "image_id", all_ids)
    probs_path = OUT_DIR / "submission_probs.csv"
    probs_df.to_csv(probs_path, index=False)
    print(f"✓ Saved per-channel max probs: {probs_path}\n")

    print("=== DONE ===")
    print(f"Upload {sub_path} to Kaggle for the no-ROI baseline.")


if __name__ == "__main__":
    main()
