"""
diagnose_roi_v2.py — Show what the new Hough-based ROI detector finds.

Run AFTER generate_submission.py to verify the detector is working correctly
on real test images. Saves a 6-image diagnostic grid showing detected ROIs.

Usage:
    python diagnose_roi_v2.py
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize bottle ROI detection on test images."
    )
    parser.parse_args()

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from krones_vision_ai.paths import ARTIFACT_ROOT, DATA_ROOT

    TEST_DIR = DATA_ROOT / "test_images"
    OUT_DIR = ARTIFACT_ROOT / "seg"
    SUB_PATH = OUT_DIR / "submission.csv"

    def detect_bottle_roi(
        img_rgb,
        blank_threshold=5.0,
        bright_threshold=80,
        min_area_frac=0.05,
        expected_radius_min=200,
        expected_radius_max=400,
        center_search_frac=0.5,
    ):
        """Same detector as in generate_submission.py — reproduced here for the diagnostic."""
        if img_rgb.ndim == 3:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_rgb
        H, W = gray.shape
        if gray.mean() < blank_threshold:
            return None, "blank"
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
            best_circle, best_score = None, -1.0
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
        _, binary = cv2.threshold(gray, bright_threshold, 255, cv2.THRESH_BINARY)
        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_lbl > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = int(np.argmax(areas)) + 1
            area = stats[largest_idx, cv2.CC_STAT_AREA]
            if (H * W * min_area_frac) < area < (H * W * 0.30):
                x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
                y = int(stats[largest_idx, cv2.CC_STAT_TOP])
                w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
                h = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
                return (x, y, w, h), "bright_blob"
        return (368, 251, 512, 512), "fallback"

    # ============================================================
    # Sample 6 images: 3 predicted Reusable, 3 predicted Not Reusable
    # Skip blank ones (already clearly handled) — pick non-blank cases
    # ============================================================
    sub = pd.read_csv(SUB_PATH)
    print(f"Loaded submission: {len(sub):,} rows")
    print(f"Predicted Reusable    : {(sub['target'] == 0).sum():,}")
    print(f"Predicted Not Reusable: {(sub['target'] == 1).sum():,}")
    print()

    # Find non-blank Reusable predictions for visualization
    non_blank_neg = []
    for fname in sub[sub["target"] == 0]["image_id"]:
        img = cv2.imread(str(TEST_DIR / fname), cv2.IMREAD_GRAYSCALE)
        if img is not None and img.mean() > 5.0:
            non_blank_neg.append(fname)
            if len(non_blank_neg) >= 3:
                break

    non_blank_pos = sub[sub["target"] == 1]["image_id"].head(3).tolist()
    samples = non_blank_neg + non_blank_pos

    _fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()

    print("ROI detection results:")
    print("-" * 90)
    print(f"{'Filename':<50}  {'pred':<5}  {'status':<12}  ROI (x,y,w,h)")
    print("-" * 90)

    for i, fname in enumerate(samples):
        img = cv2.imread(str(TEST_DIR / fname), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi, status = detect_bottle_roi(img)
        pred = sub[sub["image_id"] == fname]["target"].values[0]

        print(f"{fname[:48]:<50}  {pred:<5}  {status:<12}  {roi}")

        axes[i].imshow(img)
        if roi is not None:
            x, y, w, h = roi
            # Draw the bounding box
            rect = plt.Rectangle(
                (x, y), w, h, edgecolor="lime", facecolor="none", linewidth=3
            )
            axes[i].add_patch(rect)
            # If Hough was used, also draw the circle
            if status == "hough":
                cx, cy = x + w // 2, y + h // 2
                r = w // 2
                circle = plt.Circle(
                    (cx, cy),
                    r,
                    edgecolor="cyan",
                    facecolor="none",
                    linewidth=2,
                    linestyle="--",
                )
                axes[i].add_patch(circle)
        axes[i].set_title(f"pred={pred}, status={status}\n{fname[:40]}", fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "roi_detection_v2_check.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"\n✓ Saved: {out_path}")

    # Summary stats: how often each detection method was used
    print("\nFull-set detection method tally:")
    status_counts = {"hough": 0, "bright_blob": 0, "fallback": 0, "blank": 0}
    n_check = min(500, len(sub))
    print(f"(scanning first {n_check} images...)")
    for fname in sub["image_id"].head(n_check):
        img = cv2.imread(str(TEST_DIR / fname), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, status = detect_bottle_roi(img)
        status_counts[status] += 1

    for status, count in status_counts.items():
        pct = 100.0 * count / n_check
        print(f"  {status:<14}: {count:>4}  ({pct:.1f}%)")

    print("\nIdeal: hough should be the dominant method (>80%) on non-blank images.")
    print("If fallback is high, the detector parameters may need tuning.")


if __name__ == "__main__":
    main()
