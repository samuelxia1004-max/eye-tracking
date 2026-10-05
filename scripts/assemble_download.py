#!/usr/bin/env python3
"""Assemble fixed-range download parts and verify the official Zenodo checksum."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


RECORDS = {
    2: {
        "size": 5_601_886_507,
        "md5": "e9671316f97e1c3ec1d86a03e0a1ab28",
        "url": "https://zenodo.org/api/records/19241964/files/experiment2.zip/content",
    },
    3: {
        "size": 7_966_970_285,
        "md5": "adbe5c4766253d6be7aba530ac820ac9",
        "url": "https://zenodo.org/api/records/19241964/files/experiment3.zip/content",
    },
}


def _digests(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def assemble(experiment: int, project: Path) -> dict[str, object]:
    expected = RECORDS[experiment]
    part_dir = project / "data" / "raw" / "bbbd" / "chunks" / f"experiment{experiment}"
    parts = sorted(part_dir.glob("part*.bin"))
    if len(parts) != 16:
        raise SystemExit(f"expected 16 parts, found {len(parts)} in {part_dir}")
    part_sizes = [part.stat().st_size for part in parts]
    if sum(part_sizes) != expected["size"]:
        raise SystemExit(
            f"part-size sum mismatch: {sum(part_sizes)} != {expected['size']}; "
            f"sizes={part_sizes}"
        )
    zip_dir = project / "data" / "raw" / "bbbd" / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    destination = zip_dir / f"experiment{experiment}.zip"
    staging = zip_dir / f"experiment{experiment}.zip.assembling"
    with staging.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(8 << 20), b""):
                    output.write(chunk)
    if staging.stat().st_size != expected["size"]:
        raise SystemExit("assembled size mismatch")
    md5, sha256 = _digests(staging)
    if md5 != expected["md5"]:
        raise SystemExit(f"official MD5 mismatch: {md5} != {expected['md5']}")
    with zipfile.ZipFile(staging) as archive:
        bad_member = archive.testzip()
        member_count = len(archive.infolist())
    if bad_member is not None:
        raise SystemExit(f"ZIP CRC failed at {bad_member}")
    staging.replace(destination)
    record = {
        "experiment": experiment,
        "source_url": expected["url"],
        "zenodo_record": "https://doi.org/10.5281/zenodo.19241964",
        "license": "CC-BY-4.0",
        "bytes": expected["size"],
        "md5": md5,
        "sha256": sha256,
        "zip_member_count": member_count,
        "zip_crc": "pass",
        "assembled_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(destination.resolve()),
        "parts_retained": True,
    }
    report = project / "provenance" / "downloads" / f"experiment{experiment}_verified.json"
    report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=int, choices=sorted(RECORDS))
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.experiment, args.project.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
