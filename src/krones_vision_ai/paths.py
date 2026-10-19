"""Project and challenge-data locations shared by command-line workflows."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "KRONES_DATA_DIR",
        PROJECT_ROOT.parent / "dataset" / "1st-krones-vision-ai-challenge",
    )
).expanduser()
ARTIFACT_ROOT = Path(os.environ.get("KRONES_ARTIFACT_DIR", PROJECT_ROOT)).expanduser()


def require_paths(paths: Mapping[str, Path]) -> None:
    """Raise a clear error when required data or model paths are unavailable."""
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required paths do not exist:\n" + "\n".join(missing))
