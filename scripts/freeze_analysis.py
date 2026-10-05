#!/usr/bin/env python3
"""Record immutable input, plan, configuration, code and test hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str, project: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(project)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    config = json.loads((project / "config" / "analysis.json").read_text(encoding="utf-8"))
    if not config["status"].startswith("frozen_before_inferential_reanalysis_"):
        raise SystemExit("analysis configuration is not marked frozen")
    acceptance = json.loads((project / "provenance" / "input_acceptance.json").read_text())
    if acceptance.get("status") != "pass" or acceptance.get("inferential_outcomes_read") is not False:
        raise SystemExit("input acceptance did not pass outcome-blindly")
    paths = [
        (project / "analysis_plan.md", "analysis_plan"),
        (project / "config" / "analysis.json", "analysis_config"),
        (project / "config" / "simulation.json", "simulation_config"),
        (project / "provenance" / "input_acceptance.json", "input_acceptance"),
        (project / "provenance" / "deviations.md", "pre_outcome_deviations"),
    ]
    paths.extend((path, "inferential_source") for path in sorted((project / "src" / "boundaryisc").glob("*.py")))
    paths.extend((path, "test_source") for path in sorted((project / "tests").glob("test_*.py")))
    records = [record(path, role, project) for path, role in paths]
    input_checksums = pd.read_csv(project / "provenance" / "input_checksums.csv").to_dict(orient="records")
    payload = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_status": config["status"],
        "public_preregistration": False,
        "seed": config["seed"],
        "inferential_outcomes_read_before_freeze": False,
        "records": records,
        "input_checksums": input_checksums,
    }
    destination = project / "provenance" / "frozen_run.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "records": len(records), "path": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
