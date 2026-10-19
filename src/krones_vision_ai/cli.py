"""Command-line dispatcher for the challenge workflows."""

from __future__ import annotations

import argparse
import runpy
import sys

COMMAND_MODULES = {
    "train": "krones_vision_ai.commands.train_no_roi",
    "train-768": "krones_vision_ai.commands.train_no_roi_768",
    "infer-roi": "krones_vision_ai.commands.generate_submission",
    "infer": "krones_vision_ai.commands.generate_submission_v5",
    "diagnose": "krones_vision_ai.commands.diagnose_failures",
    "diagnose-roi": "krones_vision_ai.commands.diagnose_roi",
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch a workflow while forwarding its arguments unchanged."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="krones-vision-ai",
        description="Run a Krones Vision AI Challenge workflow.",
    )
    parser.add_argument("command", choices=COMMAND_MODULES)
    if not arguments:
        parser.print_help()
        return
    if arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
        return
    if arguments[0] not in COMMAND_MODULES:
        parser.parse_args(arguments)
        return

    original_argv = sys.argv
    try:
        sys.argv = [f"krones-vision-ai {arguments[0]}", *arguments[1:]]
        runpy.run_module(
            COMMAND_MODULES[arguments[0]],
            run_name="__main__",
            alter_sys=True,
        )
    finally:
        sys.argv = original_argv
