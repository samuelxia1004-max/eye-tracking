#!/usr/bin/env python3
"""Verify registry metadata against Crossref and preserve raw responses."""

from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover - system CA store remains the fallback
    certifi = None


def _normalized(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _normalized_field(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return _normalized(text)


def _first(message: dict, key: str) -> str:
    values = message.get(key) or []
    return str(values[0]) if values else ""


def _year(message: dict) -> int | None:
    for key in ("published-print", "published", "published-online", "issued"):
        parts = ((message.get(key) or {}).get("date-parts") or [])
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def _authors(message: dict) -> list[str]:
    return [
        " ".join(part for part in (author.get("given", ""), author.get("family", "")) if part)
        for author in message.get("author", [])
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mail-to", default="research-audit@example.invalid")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    raw = output / "crossref_raw"
    raw.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.registry)
    rows = []
    for record in registry.itertuples(index=False):
        doi = str(record.doi).strip()
        endpoint = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
        request = urllib.request.Request(
            endpoint,
            headers={"User-Agent": f"boundaryisc-reference-audit/0.2 (mailto:{args.mail_to})"},
        )
        status = "unverifiable"
        issues: list[str] = []
        try:
            context = ssl.create_default_context(cafile=certifi.where() if certifi else None)
            with urllib.request.urlopen(request, timeout=30, context=context) as response:
                payload = json.load(response)
            (raw / f"{record.source_id}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            message = payload["message"]
            crossref_title = _first(message, "title")
            crossref_authors = _authors(message)
            registry_authors = [part.strip() for part in str(record.authors).split("|")]
            if _normalized(crossref_title) != _normalized(record.title):
                issues.append("title_mismatch")
            if crossref_authors:
                crossref_families = [_normalized(value.split()[-1]) for value in crossref_authors]
                registry_families = [_normalized(value.split()[-1]) for value in registry_authors]
                if crossref_families != registry_families:
                    issues.append("author_order_or_completeness_mismatch")
            else:
                issues.append("crossref_authors_missing_check_pubmed")
            crossref_year = _year(message)
            if crossref_year is not None and int(record.year) != crossref_year:
                issues.append("publication_year_mismatch")
            for key, registered in (
                ("volume", record.volume),
                ("issue", record.issue),
                ("page", record.pages_or_article),
            ):
                crossref_value = message.get(key, "")
                if key == "page" and not crossref_value:
                    crossref_value = message.get("article-number", "")
                if pd.notna(registered) and _normalized_field(crossref_value) != _normalized_field(registered):
                    issues.append(f"{key}_mismatch")
            status = "verified" if not issues else "check_suggested"
            rows.append(
                {
                    "source_id": record.source_id,
                    "doi": doi,
                    "status": status,
                    "issues": "|".join(issues),
                    "crossref_title": crossref_title,
                    "crossref_authors": " | ".join(crossref_authors),
                    "crossref_year": crossref_year,
                    "crossref_journal": _first(message, "container-title"),
                    "crossref_volume": message.get("volume", ""),
                    "crossref_issue": message.get("issue", ""),
                    "crossref_page_or_article": message.get("page", message.get("article-number", "")),
                    "crossref_type": message.get("type", ""),
                    "is_referenced_by_count": message.get("is-referenced-by-count"),
                    "raw_response": str((raw / f"{record.source_id}.json").resolve()),
                }
            )
        except Exception as exc:  # network/metadata failures remain visible
            rows.append(
                {
                    "source_id": record.source_id,
                    "doi": doi,
                    "status": status,
                    "issues": f"{type(exc).__name__}: {exc}",
                }
            )
        time.sleep(0.1)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "reference_verification.csv", index=False)
    summary = {
        "records": int(len(frame)),
        "status_counts": frame["status"].value_counts(dropna=False).to_dict(),
        "critical_rule": "title or author mismatch requires manual resolution before citation export",
    }
    (output / "reference_verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
