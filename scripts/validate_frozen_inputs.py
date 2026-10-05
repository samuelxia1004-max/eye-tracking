#!/usr/bin/env python3
"""Hard-stop validation for BBBD and Sherlock inputs before freeze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))
    from boundaryisc.bbbd import scan_bbbd_ocular_archive

    config = json.loads((project / "config" / "analysis.json").read_text(encoding="utf-8"))
    checks: dict[str, object] = {}
    for experiment_text, archive_text in config["inputs"]["bbbd_archives"].items():
        experiment = int(experiment_text)
        manifest = scan_bbbd_ocular_archive(archive_text, experiment)
        eligible = manifest.loc[
            manifest["complete_modalities"]
            & manifest["complete_validity_masks"]
            & manifest["condition"].notna()
        ]
        session_count = eligible.groupby(["participant", "video"])["condition"].nunique()
        paired = session_count[session_count == 2].reset_index()
        actual = {
            "participants": int(manifest["participant"].nunique()),
            "paired_participants": int(paired["participant"].nunique()),
            "paired_units": int(len(paired)),
            "videos": int(manifest["video"].nunique()),
            "complete_masks": bool(manifest["complete_validity_masks"].all()),
        }
        expected = {
            "participants": int(config["bbbd"]["expected_published_included"][experiment_text]),
            "paired_participants": int(config["bbbd"]["expected_paired_participants"][experiment_text]),
            "paired_units": int(config["bbbd"]["expected_paired_participant_video_units"][experiment_text]),
            "videos": int(config["bbbd"]["expected_videos"][experiment_text]),
            "complete_masks": True,
        }
        if actual != expected:
            raise SystemExit(f"BBBD experiment {experiment} mismatch: {actual} != {expected}")
        checks[f"bbbd_experiment{experiment}"] = actual

    sherlock = np.load(config["inputs"]["sherlock_cache"], allow_pickle=False)
    groups = sherlock["group"].astype(str)
    group_counts = pd.Series(groups).value_counts().to_dict()
    sherlock_actual = {
        "participants": int(sherlock["data"].shape[0]),
        "samples": int(sherlock["data"].shape[1]),
        "channels": sherlock["channel"].astype(str).tolist(),
        "groups": {key: int(value) for key, value in group_counts.items()},
        "boundaries": len(config["sherlock"]["boundaries_10hz"]),
        "finite": bool(np.isfinite(sherlock["data"]).all()),
    }
    sherlock_expected = {
        "participants": int(config["sherlock"]["expected_participants"]),
        "samples": 12900,
        "channels": config["channels"],
        "groups": config["sherlock"]["expected_groups"],
        "boundaries": int(config["sherlock"]["expected_boundaries"]),
        "finite": True,
    }
    if sherlock_actual != sherlock_expected:
        raise SystemExit(f"Sherlock mismatch: {sherlock_actual} != {sherlock_expected}")
    checks["sherlock"] = sherlock_actual
    result = {"status": "pass", "checks": checks, "inferential_outcomes_read": False}
    destination = project / "provenance" / "input_acceptance.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
