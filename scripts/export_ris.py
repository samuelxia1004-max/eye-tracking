#!/usr/bin/env python3
"""Export the verified literature registry to ordered RIS."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _clean_number(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.registry], ignore_index=True)
    blocks = []
    seen: set[str] = set()
    for record in frame.itertuples(index=False):
        doi = str(record.doi).strip().lower()
        if doi in seen:
            raise ValueError(f"duplicate DOI: {doi}")
        seen.add(doi)
        record_type = "DATA" if str(record.record_type) == "dataset" else "JOUR"
        lines = [f"TY  - {record_type}"]
        for author in str(record.authors).split("|"):
            lines.append(f"AU  - {author.strip()}")
        lines.extend(
            [
                f"TI  - {record.title}",
                f"PY  - {int(record.year)}",
                f"JO  - {record.journal_or_repository}",
            ]
        )
        for tag, value in (
            ("VL", _clean_number(record.volume)),
            ("IS", _clean_number(record.issue)),
            ("SP", _clean_number(record.pages_or_article)),
            ("DO", doi),
            ("UR", record.official_url),
        ):
            if value and not pd.isna(value):
                lines.append(f"{tag}  - {value}")
        lines.extend([f"ID  - {record.source_id}", "ER  - "])
        blocks.append("\n".join(lines))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"wrote {len(blocks)} unique DOI records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
