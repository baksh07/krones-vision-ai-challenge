# Krones Vision AI Challenge — Project Context for Claude Code

## The competition

- 1st Krones Vision AI Challenge (Kaggle competition for our university course).
- Submission deadline: 2026-06-15. Presentation: 2026-06-27.
- Task: binary classification of empty beer-bottle base images — Reusable (0) or Not Reusable (1).
- Dataset: 35,342 train images, 4,418 test images, all 1280×1024 PNG.
- Three bottle types (0.33 L Vichy, 0.5 L Euro, 0.5 L NRW) — all in same resolution, no per-image bottle-type label.
- COCO-format annotations: 27 categories (26 defects + 1 ROI), 134,471 polygon annotations across the train set.

## Scoring

- Final score = 0.50 × F1 (hidden test set) + 0.30 × Efficiency (percentile-normalized runtime in a standard Kaggle notebook) + 0.20 × Insight (panel-judged technical insight).
- The Linatronic AI production line processes ~20 bottles/sec → 50 ms/image is our efficiency target.
- Prizes: 1st €1500, 2nd €1000, 3rd €500.

## Strategic decision: business-rule-native segmentation (NOT YOLO)

Most teams will throw YOLO at this. Our differentiation: train a multi-class semantic segmentation model on the COCO polygons, then apply the published area thresholds per defect class as deterministic post-processing to derive the binary label. This gives auditability, adapts to threshold changes without retraining, and produces a strong Insight slide.

## The published business rules

- 4 "good" categories (never count as defects): Embossing, Foam residue, No fault, Water drop.
- 6 "conditional" categories (defect only if area > threshold):
  - Air bubble > 500 px
  - Chip > 200 px
  - Contamination light > 180 px
  - Glass imperfection > 100 px
  - Scuffing > 75,000 px
  - Scuffing heavy > 1,200 px
- 16 "always faulty" categories (any pixel = defect): Break/Crack, Circlip, Contamination dark, Crown cap, Foil/Semitransparent, Foreign object - manual cleaning, Foreign object - washing machine, Glass shard, Insect, Label, Liquid, Mold, No base visible, Paint residue, Straw, Yeast residue.
- Image-level rule: defect if ANY annotation triggers (per-region, NOT aggregated by default).

## EDA findings (from notebook 01, already run)

- 0.19% disagreement rate between the rules-derived label and the official train.csv label — extremely clean dataset.
- All 68 disagreements go in one direction: official=1 but rules=0. **Zero disagreements in the other direction.**
- 100% of the 68 disagreements involve Scuffing — multiple sub-threshold blobs on one bottle.
- Hypothesis: production line aggregates Scuffing area across blobs, while published rule is per-region.
- Class balance: 58% Not Reusable / 42% Reusable. Mild imbalance, no special handling needed.
- UUID prefix in filenames is unique per image (not a session identifier) — splits are simple stratified KFold.
- Every image is exactly 1280×1024, so the area-scale factor for 384×384 input is the constant (1280×1024)/(384²) ≈ 8.8889.

### Per-category trigger reality (key insight)

| Category | Annotations | % triggering threshold | Practical meaning |
|---|---|---|---|
| Chip | 717 | 99.7% | effectively binary |
| Glass imperfection | 180 | 96.1% | effectively binary, also rare |
| Contamination light | 1,956 | 45.1% | true threshold behavior |
| Scuffing heavy | 765 | 42.2% | true threshold behavior |
| Scuffing | 10,058 | 4.6% | almost always background |
| Air bubble | 3,427 | 0.17% | almost always harmless |

## Server environment

- University server: `dll0202@DEGGS111PC02` (10.60.5.2)
- GPU: NVIDIA RTX A5000, 24 GB VRAM, CUDA 12.2 driver, 12.1 PyTorch runtime
- Python 3.10.12, venv at `~/baksh/krones/venv/` — ALWAYS activate before running anything: `source ~/baksh/krones/venv/bin/activate`
- Installed: torch 2.4.1+cu121, torchvision 0.19.1, segmentation-models-pytorch 0.3.4, albumentations 1.4.18, opencv-python-headless 4.13.0, pandas, sklearn, matplotlib, tqdm, pycocotools, jupyter, ipykernel.
- Jupyter kernel registered as "Krones (venv)" — select this in VS Code top-right when opening notebooks.
- Sudo: NOT available passwordless. Don't try to apt-install anything; use pip in the venv.

## File layout
~/baksh/
├── dataset/
│   └── 1st-krones-vision-ai-challenge/
│       ├── train_images/   (35,342 PNGs, 1280×1024)
│       ├── test_images/    (4,418 PNGs)
│       ├── train.csv
│       ├── sample_submission.csv
│       └── train_annotations.json   (COCO format)
└── krones/
├── venv/                              # Python virtual env
├── 01_eda_and_split (1).ipynb         # already run, produced outputs below
├── train_enriched.csv                 # 35,342 rows: target, derived_target, session_uuid, fold, etc.
├── label_noise_candidates.csv         # the 68 disagreements
├── category_stats.csv                 # per-category frequency + area distribution
├── eda_overview.png                   # presentation-ready chart
├── splits/
│   ├── fold_0.csv                     # PRIMARY validation set — every F1 number from this
│   ├── fold_1.csv ... fold_4.csv      # for optional 5-fold ensembling later
├── 02_segmentation_training_server.ipynb   # ready to run (see below)
└── seg/                               # output dir for notebook 02 (created at runtime)

## The current task: notebook 02

`~/baksh/krones/02_segmentation_training_server.ipynb` is configured for this server:
- Paths set to `/home/dll0202/baksh/...`
- batch_size=32, num_workers=8, lr=4e-4 (linear-scaled for the bigger batch)
- Architecture: U-Net++ with EfficientNet-B0 encoder (Severstal-winning recipe)
- Output: 8 channels — one per "decision class" (6 conditional + 1 always_faulty + 1 auxiliary 'harmless')
- Loss: weighted Dice + BCE, channel weights boost rare classes (e.g. glass_imperfection × 3)
- Augmentations (albumentations 1.x API): HFlip, ShiftScaleRotate, Brightness/Contrast, Gamma, GaussNoise, CoarseDropout — all chosen to attack artifact-confusion (water drops/foam vs real defects)

### Key innovation: `Cfg.aggregate_scuffing` flag (default False)

The decision function `predict_binary_from_logits` takes a `aggregated_channels: set[int]` parameter. When `Cfg.aggregate_scuffing=True`, the Scuffing & Scuffing-heavy channels use SUM of all blob areas instead of MAX — capturing the 68 disagreement cases identified in EDA. This is a critical ablation: train ONCE, then evaluate the same model under both decision rules to get F1_rule_faithful vs F1_aggregated. The Insight slide is "we identified the discrepancy in EDA, then quantified the impact."

## What the user wants next

1. Smoke test first: `CFG.epochs=2, CFG.img_size=256, CFG.batch_size=16` — confirm pipeline works end-to-end (~10 min).
2. If smoke test gives val F1 > 0.5, run the full thing: `CFG.epochs=25, CFG.img_size=384, CFG.batch_size=32` (~2-3 hours on A5000).
3. After training: load the checkpoint, run evaluate() twice (with `aggregate_scuffing` False then True), compare F1s.
4. Send me back: best val F1 for both rules + per-channel precision summary from `seg/channel_precision.csv`.

## What to do for the user

- Activate venv before any python invocations.
- If a cell fails, read the traceback first — the most common failures are: path issues (assertions will catch), albumentations API mismatch (we pinned 1.4.18 to avoid this), or OOM (drop batch_size to 16).
- Don't bypass the path assertions — they're there to fail fast.
- The user is a student preparing for a competition presentation. Insight stories matter as much as raw F1. When you find something interesting in the data or training, flag it as potential presentation material.
- After training: ALWAYS run the per_channel_audit cell. If channels with high loss-weight (especially glass_imperfection) show 0 triggers, the model didn't learn them — boost their weight further.

## Communication style preferred

- Be direct and substantive, not chatty.
- Show the user what you're doing (run the cell, paste the output).
- When the user asks "what's happening?" they want a real explanation, not just code.
- This project is collaborative — proactively suggest next experiments and tradeoffs.

