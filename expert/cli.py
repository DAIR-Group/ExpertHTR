"""Command-line interface for the ExpertHTR reproducibility workflow."""

from __future__ import annotations

import argparse
from .workflow import available_commands, phase_names_for, run, validate_phase_imports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experthtr",
        description="Run the released ExpertHTR dense-to-MoE workflow without a notebook.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=available_commands(),
        help="Workflow command to execute.",
    )
    parser.add_argument(
        "--print-phases",
        action="store_true",
        help="Print the ordered named Python phases for COMMAND and exit.",
    )
    parser.add_argument(
        "--validate-imports",
        action="store_true",
        help="Validate Python phase registration without loading model or data and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.validate_imports:
        phases = validate_phase_imports()
        print(f"PASS: registered {len(phases)} named ExpertHTR Python phases")
        return 0
    if args.command is None:
        parser.error("COMMAND is required unless --validate-imports is used.")
    if args.print_phases:
        print("\n".join(phase_names_for(args.command)))
        return 0

    run(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
