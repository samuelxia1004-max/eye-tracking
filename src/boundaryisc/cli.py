"""Command-line interface for the frozen BBBD-to-Sherlock study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .study_pipeline import (
    prepare_bbbd_cache,
    run_bbbd_benchmark,
    run_bbbd_threshold_sensitivity,
    run_dual_null_simulations,
    run_eeg_convergence,
    run_sherlock_transport,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boundaryisc")
    parser.add_argument("--config", default="config/analysis.json")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-config", "reproduce", "analyze", "simulate", "report"):
        sub.add_parser(command)
    return parser


def _load(path: str) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project = Path(config.get("project_root", config_path.parent.parent)).resolve()
    return config, project


def _validate(config: dict, project: Path) -> dict[str, object]:
    required = {"seed", "rate_hz", "window_seconds", "bbbd", "sherlock", "inference", "inputs"}
    missing = required.difference(config)
    if missing:
        raise SystemExit(f"configuration is incomplete: {sorted(missing)}")
    if not str(config.get("status", "")).startswith("frozen"):
        raise SystemExit("inferential commands require a frozen configuration")
    input_paths = [
        *[Path(path) for path in config["inputs"]["bbbd_archives"].values()],
        Path(config["inputs"]["sherlock_cache"]),
    ]
    absent = [str(path) for path in input_paths if not path.exists()]
    if absent:
        raise SystemExit(f"frozen input is absent: {absent}")
    return {
        "project": str(project),
        "status": config["status"],
        "seed": config["seed"],
        "inputs_present": len(input_paths),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config, project = _load(args.config)
    if args.command == "validate-config":
        print(json.dumps(_validate(config, project), indent=2))
        return 0
    _validate(config, project)
    if args.command == "reproduce":
        path = prepare_bbbd_cache(config, project)
        print(json.dumps({"bbbd_cache": str(path)}, indent=2))
    elif args.command == "analyze":
        benchmark = run_bbbd_benchmark(config, project)
        sensitivity = run_bbbd_threshold_sensitivity(config, project)
        eeg = run_eeg_convergence(config, project)
        transport = run_sherlock_transport(config, project)
        print(json.dumps({"benchmark": benchmark, "sensitivity": sensitivity, "eeg": eeg, "transport": transport}, indent=2))
    elif args.command == "simulate":
        print(json.dumps(run_dual_null_simulations(config, project), indent=2))
    elif args.command == "report":
        from .reporting import build_report_artifacts

        print(json.dumps(build_report_artifacts(config, project), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
