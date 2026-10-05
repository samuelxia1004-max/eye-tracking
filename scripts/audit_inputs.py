#!/usr/bin/env python3
"""Create outcome-free input manifests before the inferential freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--sherlock-project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))
    from boundaryisc.bbbd import scan_bbbd_ocular_archive

    rows = []
    manifests = []
    for experiment in (2, 3):
        archive = project / "data" / "raw" / "bbbd" / "zips" / f"experiment{experiment}.zip"
        manifest = scan_bbbd_ocular_archive(archive, experiment)
        manifests.append(manifest)
        rows.append(
            {
                "role": f"bbbd_experiment{experiment}",
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
            }
        )
    bbbd_manifest = pd.concat(manifests, ignore_index=True)
    bbbd_manifest.to_csv(project / "provenance" / "bbbd_ocular_archive_manifest.csv", index=False)
    summary_rows = []
    for experiment, block in bbbd_manifest.groupby("experiment", sort=True):
        eligible = block.loc[
            block["complete_modalities"]
            & block["complete_validity_masks"]
            & block["condition"].notna()
        ]
        session_count = eligible.groupby(["participant", "video"])["condition"].nunique()
        paired = session_count[session_count == 2]
        summary_rows.append(
            {
                "experiment": int(experiment),
                "participants": int(block["participant"].nunique()),
                "videos": int(block["video"].nunique()),
                "sessions": int(block["session"].nunique()),
                "records": int(len(block)),
                "complete_records": int(block["complete_modalities"].sum()),
                "complete_validity_masks": int(block["complete_validity_masks"].sum()),
                "paired_participant_video_units": int(len(paired)),
                "paired_records": int(2 * len(paired)),
                "participants_with_pair": int(paired.reset_index()["participant"].nunique()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(project / "provenance" / "bbbd_ocular_archive_summary.csv", index=False)

    sherlock_files = {
        "sherlock_cache": args.sherlock_project / "derived" / "commercial_10hz.npz",
        "sherlock_participant_manifest": args.sherlock_project / "provenance" / "participant_manifest.csv",
        "sherlock_frozen_run": args.sherlock_project / "provenance" / "frozen_run.json",
    }
    for role, path in sherlock_files.items():
        path = path.resolve()
        rows.append(
            {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    checksum_frame = pd.DataFrame(rows)
    checksum_frame.to_csv(project / "provenance" / "input_checksums.csv", index=False)

    candidate = {
        "status": "candidate_ready_for_manual_freeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bbbd_summary": summary.to_dict(orient="records"),
        "input_checksums": checksum_frame.to_dict(orient="records"),
        "no_outcome_statistics_computed": True,
    }
    (project / "provenance" / "freeze_candidate.json").write_text(
        json.dumps(candidate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(candidate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
