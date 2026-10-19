"""Project and challenge-data locations shared by command-line workflows."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "KRONES_DATA_DIR",
        PROJECT_ROOT.parent / "dataset" / "1st-krones-vision-ai-challenge",
    )
).expanduser()
ARTIFACT_ROOT = Path(
    os.environ.get("KRONES_ARTIFACT_DIR", PROJECT_ROOT)
).expanduser()
