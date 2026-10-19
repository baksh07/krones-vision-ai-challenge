from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

from krones_vision_ai import cli, rules


def test_decision_rules_preserve_channel_order_and_thresholds() -> None:
    assert rules.CHANNEL_NAMES == [
        "air_bubble",
        "chip",
        "contamination_light",
        "glass_imperfection",
        "scuffing",
        "scuffing_heavy",
        "always_faulty",
        "harmless",
    ]
    assert rules.CHANNEL_THRESHOLDS == [500, 200, 180, 100, 75_000, 1_200, 1, None]
    assert rules.AGGREGATED_CHANNEL_INDICES == frozenset({4, 5})
    assert "No fault" not in rules.DECISION_CHANNELS["harmless"]["cats"]


def test_command_modules_are_safe_to_import() -> None:
    for module_name in cli.COMMAND_MODULES.values():
        module = importlib.import_module(module_name)
        assert callable(module.main)


def test_cli_forwards_arguments_and_restores_argv() -> None:
    original_argv = sys.argv
    forwarded_argv: list[str] = []

    def run_module(module_name: str, **_: object) -> None:
        forwarded_argv.extend(sys.argv)
        assert module_name == "krones_vision_ai.commands.diagnose_failures"

    with patch.object(cli.runpy, "run_module", side_effect=run_module):
        cli.main(["diagnose", "--ckpt", "seg/best.pt"])

    assert forwarded_argv == [
        "krones-vision-ai diagnose",
        "--ckpt",
        "seg/best.pt",
    ]
    assert sys.argv is original_argv


def test_data_and_artifact_paths_can_be_configured(
    monkeypatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "challenge-data"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("KRONES_DATA_DIR", str(data_dir))
    monkeypatch.setenv("KRONES_ARTIFACT_DIR", str(artifact_dir))

    paths = importlib.reload(importlib.import_module("krones_vision_ai.paths"))
    assert paths.DATA_ROOT == data_dir
    assert paths.ARTIFACT_ROOT == artifact_dir
