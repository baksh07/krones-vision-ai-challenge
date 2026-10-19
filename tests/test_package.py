from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from krones_vision_ai import cli, rules
from krones_vision_ai.decisions import predict_from_logits
from krones_vision_ai.paths import require_paths


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


def test_cli_shows_help_when_no_command_is_given(capsys) -> None:
    cli.main([])

    assert "Run a Krones Vision AI Challenge workflow." in capsys.readouterr().out


def test_cli_forwards_subcommand_help() -> None:
    forwarded_argv: list[str] = []

    def run_module(module_name: str, **_: object) -> None:
        forwarded_argv.extend(sys.argv)
        assert module_name == "krones_vision_ai.commands.diagnose_failures"

    with patch.object(cli.runpy, "run_module", side_effect=run_module):
        cli.main(["diagnose", "--help"])

    assert forwarded_argv == ["krones-vision-ai diagnose", "--help"]


@pytest.mark.parametrize("module_name", cli.COMMAND_MODULES.values())
def test_command_help_is_available_without_vision_dependencies(
    module_name: str, capsys
) -> None:
    command = importlib.import_module(module_name)

    with (
        patch.object(sys, "argv", [module_name, "--help"]),
        pytest.raises(SystemExit) as error,
    ):
        command.main()

    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_prediction_rules_support_component_aggregation_and_audit_triggers() -> None:
    logits = torch.full((1, 3, 4, 4), -10.0)
    logits[0, 0, 0, 0] = 10.0
    logits[0, 0, 3, 3] = 10.0
    logits[0, 1, :2, :2] = 10.0
    logits[0, 2, :, :] = 10.0

    prediction = predict_from_logits(
        logits,
        [1, 2, None],
        area_scale=1.0,
        aggregated_channels={0},
    )

    assert prediction.predictions.tolist() == [1]
    assert prediction.triggers == (((0, 2), (1, 4)),)
    np.testing.assert_allclose(prediction.max_probabilities, [[1, 1, 1]], atol=1e-4)


def test_prediction_threshold_is_strict_and_keeps_max_blob_semantics() -> None:
    logits = torch.full((1, 1, 4, 4), -10.0)
    logits[0, 0, 0, 0] = 10.0
    logits[0, 0, 3, 3] = 10.0

    prediction = predict_from_logits(logits, [1])

    assert prediction.predictions.tolist() == [0]
    assert prediction.triggers == ((),)


@pytest.mark.parametrize(
    ("logits", "thresholds", "options", "message"),
    [
        (torch.zeros(1, 2, 2), [1], {}, "BxCxHxW"),
        (torch.zeros(1, 2, 2, 2), [1], {}, "logit channels"),
        (torch.zeros(1, 1, 2, 2), [1], {"area_scale": 0}, "area_scale"),
        (
            torch.zeros(1, 1, 2, 2),
            [1],
            {"aggregated_channels": {1}},
            "channel index",
        ),
    ],
)
def test_prediction_rejects_invalid_inputs(
    logits: torch.Tensor,
    thresholds: list[int | None],
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        predict_from_logits(logits, thresholds, **options)


def test_data_and_artifact_paths_can_be_configured(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "challenge-data"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("KRONES_DATA_DIR", str(data_dir))
    monkeypatch.setenv("KRONES_ARTIFACT_DIR", str(artifact_dir))

    paths = importlib.reload(importlib.import_module("krones_vision_ai.paths"))
    assert paths.DATA_ROOT == data_dir
    assert paths.ARTIFACT_ROOT == artifact_dir


def test_required_paths_report_all_missing_inputs(tmp_path: Path) -> None:
    missing_image_dir = tmp_path / "images"
    missing_annotations = tmp_path / "annotations.json"

    with pytest.raises(FileNotFoundError) as error:
        require_paths(
            {
                "training images": missing_image_dir,
                "annotations": missing_annotations,
            }
        )

    message = str(error.value)
    assert "training images" in message
    assert "annotations" in message
