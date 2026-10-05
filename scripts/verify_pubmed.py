#!/usr/bin/env python3
"""Cross-check DOI-linked registry records against official PubMed summaries."""

from __future__ import annotations

import argparse
import json
import re
import ssl
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


def _normalized(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _surname_from_registry(name: str) -> str:
    return _normalized(name.strip().split()[-1])


def _surname_from_pubmed(name: str) -> str:
    return _normalized(name.strip().split()[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.registry)
    indexed = frame.loc[frame["pmid"].notna()].copy()
    indexed["pmid_text"] = indexed["pmid"].astype(int).astype(str)
    query = urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(indexed["pmid_text"]), "retmode": "json"}
    )
    endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + query
    context = ssl.create_default_context(cafile=certifi.where() if certifi else None)
    with urllib.request.urlopen(endpoint, timeout=60, context=context) as response:
        payload = json.load(response)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "pubmed_esummary_raw.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = []
    result = payload["result"]
    for record in indexed.itertuples(index=False):
        item = result[record.pmid_text]
        issues = []
        if _normalized(item.get("title", "").rstrip(".")) != _normalized(record.title):
            issues.append("title_mismatch")
        pubmed_surnames = [
            _surname_from_pubmed(author.get("name", "")) for author in item.get("authors", [])
        ]
        registry_surnames = [
            _surname_from_registry(author) for author in str(record.authors).split("|")
        ]
        if pubmed_surnames != registry_surnames:
            issues.append("author_order_or_completeness_mismatch")
        doi_values = [
            entry.get("value", "").lower()
            for entry in item.get("articleids", [])
            if entry.get("idtype") == "doi"
        ]
        if str(record.doi).lower() not in doi_values:
            issues.append("doi_mismatch_or_missing")
        publication_types = item.get("pubtype", []) or []
        retraction_flags = [
            value
            for value in publication_types
            if "retract" in value.casefold() or "withdraw" in value.casefold()
        ]
        if retraction_flags:
            issues.append("retraction_or_withdrawal_flag")
        rows.append(
            {
                "source_id": record.source_id,
                "pmid": record.pmid_text,
                "doi": record.doi,
                "status": "verified" if not issues else "needs_review",
                "issues": "|".join(issues),
                "pubmed_title": item.get("title", ""),
                "pubmed_authors": " | ".join(author.get("name", "") for author in item.get("authors", [])),
                "pubmed_date": item.get("pubdate", ""),
                "pubmed_journal": item.get("fulljournalname", ""),
                "pubmed_volume": item.get("volume", ""),
                "pubmed_issue": item.get("issue", ""),
                "pubmed_pages": item.get("pages", ""),
                "publication_types": " | ".join(publication_types),
                "retraction_flags": " | ".join(retraction_flags),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "pubmed_verification.csv", index=False)
    summary = {
        "records_with_pmid": int(len(audit)),
        "status_counts": audit["status"].value_counts().to_dict(),
        "retraction_or_withdrawal_flags": int(audit["retraction_flags"].astype(bool).sum()),
    }
    (output / "pubmed_verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
