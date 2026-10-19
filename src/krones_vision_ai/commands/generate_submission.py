"""
generate_submission.py — Krones Vision AI Challenge

Loads ~/baksh/krones/seg/best.pt and produces submission.csv
by running inference on all 4,418 test images.

Usage (from ~/baksh/krones, with venv activated):
    python -m krones_vision_ai infer-roi

The script reads:
    - best.pt              (checkpoint with model_state, cfg, channel_thresholds)
    - dataset/.../test_images/*.png
    - dataset/.../sample_submission.csv  (only used to verify our id list matches)

The script writes:
    - seg/submission.csv         (4,418 rows: image_id, target)
    - seg/submission_probs.csv   (per-channel max probabilities — for debugging)
"""

from __future__ import annotations


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a test submission with the ROI-based model."
    )
    parser.parse_args()

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

    from krones_vision_ai.paths import ARTIFACT_ROOT, DATA_ROOT

    # ============================================================================
    # Paths — adjust if your layout differs
    # ============================================================================
    DATA_DIR = DATA_ROOT
    ART_DIR = ARTIFACT_ROOT
    OUT_DIR = ART_DIR / "seg"
    CKPT_PATH = OUT_DIR / "best.pt"

    TEST_IMG_DIR = DATA_DIR / "test_images"
    SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

    from krones_vision_ai.paths import require_paths

    require_paths(
        {
            "checkpoint": CKPT_PATH,
            "test images": TEST_IMG_DIR,
            "sample submission": SAMPLE_SUB,
        }
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ============================================================================
    # Channel definitions — must match what the model was trained on
    # ============================================================================
    from krones_vision_ai.rules import (
        AGGREGATED_CHANNEL_INDICES,
        CHANNEL_NAMES,
        CHANNEL_THRESHOLDS,
        N_CHANNELS,
    )

    # ============================================================================
    # Load checkpoint and reconstruct config
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

    @dataclass
    class InferCfg:
        img_size: int = saved_cfg["img_size"]
        encoder_name: str = saved_cfg["encoder_name"]
        encoder_weights: str = saved_cfg["encoder_weights"]
        arch: str = saved_cfg["arch"]
        orig_h: int = saved_cfg.get("orig_h", 1024)
        orig_w: int = saved_cfg.get("orig_w", 1280)
        batch_size: int = 32
        num_workers: int = 4
        aggregate_scuffing: bool = bool(ckpt.get("aggregate_scuffing", False))

        @property
        def area_scale(self) -> float:
            return (self.orig_h * self.orig_w) / (self.img_size * self.img_size)

    CFG = InferCfg()
    CFG.batch_size = 16  # safe inference batch — was hitting OOM at 32
    print(f"  area_scale (derived): {CFG.area_scale:.4f}")
    print(f"  batch_size (overridden for inference): {CFG.batch_size}")

    # ============================================================================
    # Build model and load weights
    # ============================================================================
    from krones_vision_ai.models import build_model

    model = build_model(CFG, n_classes=N_CHANNELS, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("  Model loaded ✓\n")

    # ============================================================================
    # Test dataset — no ROI crop, no augmentation
    # ============================================================================
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def build_test_aug(size: int) -> A.Compose:
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
    # Auto-detect bottle ROI in a test image (v2 — Hough circles with fallbacks)
    # ============================================================================
    # The bottle BASE is the bright disc in the center of every image. Naive
    # brightness thresholding fails because the surrounding glass cylinder is also
    # bright (its reflections merge with the base into one big "bright blob").
    #
    # Strategy:
    #   1. Blank check (mean brightness too low → no bottle, return verdict 0).
    #   2. Hough circle detection with tight constraints:
    #      - radius in expected range (200..400 px, the bottle base size)
    #      - center within 50% of image center
    #      - mean brightness inside circle > 100 (the base IS bright)
    #      Score by brightness × radius² and pick best.
    #   3. Bright-blob fallback (legacy method, but stricter — rejects blobs > 30%
    #      of image, since those are the cylinder, not the base).
    #   4. Ultimate fallback: median train-set ROI (368, 251, 512, 512).
    #
    # Returns: (roi_tuple_or_None, status_string)
    #   roi_tuple = (x, y, w, h) or None
    #   status    = "hough" | "bright_blob" | "fallback" | "blank"
    # ============================================================================
    def detect_bottle_roi(
        img_rgb: np.ndarray,
        blank_threshold: float = 5.0,
        bright_threshold: int = 80,
        min_area_frac: float = 0.05,
        expected_radius_min: int = 200,
        expected_radius_max: int = 400,
        center_search_frac: float = 0.5,
    ) -> tuple[tuple[int, int, int, int] | None, str]:
        """Detect the bright bottle base region using Hough circles + fallbacks."""
        if img_rgb.ndim == 3:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_rgb
        H, W = gray.shape

        # Step 1: blank-image short-circuit
        if gray.mean() < blank_threshold:
            return None, "blank"

        # Step 2: Hough Circle Transform — the proper way
        gray_blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=W // 2,
            param1=80,
            param2=30,
            minRadius=expected_radius_min,
            maxRadius=expected_radius_max,
        )
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            cx_img, cy_img = W // 2, H // 2
            center_tol = int(min(W, H) * center_search_frac / 2)
            best_circle = None
            best_score = -1.0
            for cx, cy, r in circles:
                if abs(cx - cx_img) > center_tol or abs(cy - cy_img) > center_tol:
                    continue
                mask = np.zeros_like(gray)
                cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
                mean_brightness = cv2.mean(gray, mask=mask)[0]
                if mean_brightness < 100:
                    continue
                score = mean_brightness * (r * r)
                if score > best_score:
                    best_score = score
                    best_circle = (cx, cy, r)
            if best_circle is not None:
                cx, cy, r = best_circle
                x = max(0, int(cx - r))
                y = max(0, int(cy - r))
                w = min(W - x, int(2 * r))
                h = min(H - y, int(2 * r))
                return (x, y, w, h), "hough"

        # Step 3: bright-blob fallback (stricter than v1 — reject huge blobs)
        _, binary = cv2.threshold(gray, bright_threshold, 255, cv2.THRESH_BINARY)
        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_lbl > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = int(np.argmax(areas)) + 1
            area = stats[largest_idx, cv2.CC_STAT_AREA]
            # Reject if blob is too small (noise) OR too big (cylinder)
            if (H * W * min_area_frac) < area < (H * W * 0.30):
                x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
                y = int(stats[largest_idx, cv2.CC_STAT_TOP])
                w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
                h = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
                return (x, y, w, h), "bright_blob"

        # Step 4: ultimate fallback — median train ROI
        return (368, 251, 512, 512), "fallback"

    class TestDataset(Dataset):
        """Test-time dataset that auto-detects and crops to the bottle ROI.

        The training pipeline cropped each image to its COCO-defined ROI before
        resizing to img_size. Test images have no COCO annotations, so we detect
        the bright circular bottle region automatically. Images that are entirely
        black (no bottle) get flagged so we can skip model inference and return
        verdict=0 directly.
        """

        def __init__(
            self,
            df: pd.DataFrame,
            img_dir: Path,
            transform: A.Compose,
            target_size: int,
        ):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transform = transform
            self.target_size = target_size

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> dict:
            row = self.df.iloc[idx]
            fname = row["image_id"]
            img_path = self.img_dir / fname
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect bottle, crop to its bounding box (with padding) — matches training
            roi, status = detect_bottle_roi(img)
            is_blank = status == "blank"

            if is_blank:
                # Use a black frame; we'll skip inference for this image anyway
                img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            elif roi is not None:
                x, y, w, h = roi
                H, W = img.shape[:2]
                pad = 10
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(W, x + w + pad)
                y1 = min(H, y + h + pad)
                if x1 > x0 and y1 > y0:
                    img = img[y0:y1, x0:x1]
            # else: detection ambiguous — use full image as fallback

            out = self.transform(image=img)
            return {"image": out["image"], "image_id": fname, "blank": is_blank}

    sample_sub = pd.read_csv(SAMPLE_SUB)
    print(f"Sample submission has {len(sample_sub):,} rows")
    print(f"Columns: {sample_sub.columns.tolist()}")
    print(f"First 3 image_ids: {sample_sub['image_id'].head(3).tolist()}\n")

    test_ds = TestDataset(
        sample_sub[["image_id"]],
        TEST_IMG_DIR,
        build_test_aug(CFG.img_size),
        CFG.img_size,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    from krones_vision_ai.decisions import predict_from_logits

    # ============================================================================
    # Run inference
    # ============================================================================
    print("=" * 60)
    print(f"Running inference on {len(test_ds):,} test images")
    print(f"  aggregate_scuffing : {CFG.aggregate_scuffing}")
    print(f"  img_size           : {CFG.img_size}")
    print(f"  batch_size         : {CFG.batch_size}")
    print("=" * 60)

    agg = AGGREGATED_CHANNEL_INDICES if CFG.aggregate_scuffing else set()
    all_ids: list[str] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    n_blank = 0

    # Critical: torch.no_grad() wraps the whole forward pass.
    # Without it, PyTorch builds the autograd graph for every batch and VRAM grows
    # until OOM. With it, no graph is built — memory stays bounded.
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        for batch in tqdm(test_loader, desc="inference", total=len(test_loader)):
            img = batch["image"].to(DEVICE, non_blocking=True)
            ids = batch["image_id"]
            blank_flags = batch["blank"]
            logits = model(img)
            prediction_batch = predict_from_logits(
                logits,
                CHANNEL_THRESHOLDS,
                area_scale=CFG.area_scale,
                aggregated_channels=agg,
            )
            verdicts = prediction_batch.predictions
            max_probs = prediction_batch.max_probabilities
            # Override verdict to 0 for blank images (no bottle present)
            for i, b in enumerate(blank_flags):
                if bool(b):
                    verdicts[i] = 0
                    n_blank += 1
            all_ids.extend(ids)
            all_preds.extend(verdicts.tolist())
            all_probs.append(max_probs)
            del logits, img

    torch.cuda.empty_cache()
    print(f"\n{n_blank} blank/black image(s) detected — set to verdict=0 directly")

    print(f"\n✓ Inference complete on {len(all_ids):,} images")

    # ============================================================================
    # Build submission DataFrame and verify
    # ============================================================================
    sub = pd.DataFrame({"image_id": all_ids, "target": all_preds})

    sample_set = set(sample_sub["image_id"])
    ours_set = set(sub["image_id"])
    assert sample_set == ours_set, (
        f"Image-id mismatch! "
        f"In sample but not ours: {len(sample_set - ours_set)}, "
        f"In ours but not sample: {len(ours_set - sample_set)}"
    )

    sub = sub.set_index("image_id").reindex(sample_sub["image_id"]).reset_index()

    n_pos = int(sub["target"].sum())
    n_neg = len(sub) - n_pos
    print("\nPrediction distribution:")
    print(f"  Reusable (0)    : {n_neg:>5,}  ({100 * n_neg / len(sub):.1f}%)")
    print(f"  Not Reusable (1): {n_pos:>5,}  ({100 * n_pos / len(sub):.1f}%)")
    print(f"  Total           : {len(sub):>5,}")

    ratio = n_pos / len(sub)
    if ratio < 0.30 or ratio > 0.85:
        print(f"\n⚠️  WARNING: ratio {ratio:.1%} differs sharply from train (58.3%).")
        print(
            "   Worth investigating — either test distribution differs or calibration issue."
        )

    # ============================================================================
    # Save outputs
    # ============================================================================
    sub_path = OUT_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n✓ Saved submission: {sub_path}")
    print(f"  Rows: {len(sub):,}")
    print(f"  Columns: {sub.columns.tolist()}")

    probs_arr = np.concatenate(all_probs, axis=0)
    probs_df = pd.DataFrame(probs_arr, columns=CHANNEL_NAMES)
    probs_df.insert(0, "image_id", all_ids)
    probs_path = OUT_DIR / "submission_probs.csv"
    probs_df.to_csv(probs_path, index=False)
    print(f"✓ Saved per-channel max probs: {probs_path}\n")

    print("First 5 rows of submission.csv:")
    print(sub.head().to_string(index=False))
    print("\n=== DONE ===")
    print(f"Upload {sub_path} to Kaggle to record your first leaderboard entry.")


if __name__ == "__main__":
    main()
