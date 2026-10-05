#!/usr/bin/env python3
"""Download lawful PMC OA packages and verify extracted article PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import ssl
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where() if certifi else None)


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "boundaryisc-lawful-oa-downloader/0.2"},
    )
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(request, timeout=180, context=_context()) as response:
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=8 << 20)
    temporary.replace(path)


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"unsafe archive member: {member.name}")
    archive.extractall(destination, filter="data")


def _pdf_pages(path: Path) -> int | None:
    if fitz is None:
        return None
    with fitz.open(path) as document:
        return int(document.page_count)


def _europe_pmc_xml(source_id: str, pmcid: str, destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    path = destination / f"{source_id}_{pmcid}.xml"
    if not path.exists():
        _download(url, path)
    root = ET.parse(path).getroot()
    title = " ".join(root.findtext(".//article-title", default="").split())
    return {
        "status": "fulltext_xml_verified",
        "fulltext_xml_url": url,
        "fulltext_xml": str(path),
        "fulltext_xml_bytes": path.stat().st_size,
        "fulltext_xml_sha256": _sha256(path),
        "xml_article_title": title,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    packages = project / "literature" / "oa_packages"
    extracted_root = packages / "extracted"
    fulltext = project / "literature" / "fulltext"
    supporting = project / "literature" / "supporting"
    fulltext_xml = project / "literature" / "fulltext_xml"
    for directory in (packages, extracted_root, fulltext, supporting, fulltext_xml):
        directory.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.registry)
    rows = []
    for record in registry.itertuples(index=False):
        pmcid = "" if pd.isna(record.pmcid) else str(record.pmcid).strip()
        base = {
            "source_id": record.source_id,
            "doi": record.doi,
            "pmcid": pmcid,
            "official_url": record.official_url,
        }
        if not pmcid:
            rows.append({**base, "status": "metadata_only_no_pmc_oa_package"})
            continue
        api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
        try:
            request = urllib.request.Request(api, headers={"User-Agent": "boundaryisc-lawful-oa-downloader/0.2"})
            with urllib.request.urlopen(request, timeout=60, context=_context()) as response:
                root = ET.fromstring(response.read())
            record_node = root.find(".//record")
            link = root.find(".//link[@format='tgz']")
            if record_node is None or link is None or not link.attrib.get("href"):
                try:
                    rows.append({**base, **_europe_pmc_xml(record.source_id, pmcid, fulltext_xml), "oa_api": api})
                except Exception as fallback_exc:
                    rows.append(
                        {
                            **base,
                            "status": "metadata_only_not_in_pmc_oa_subset",
                            "oa_api": api,
                            "error": f"{type(fallback_exc).__name__}: {fallback_exc}",
                        }
                    )
                continue
            package_url = re.sub(r"^ftp://", "https://", link.attrib["href"])
            package_path = packages / f"{record.source_id}_{pmcid}.tar.gz"
            if not package_path.exists():
                _download(package_url, package_path)
            extracted = extracted_root / record.source_id
            if not extracted.exists():
                extracted.mkdir(parents=True)
                with tarfile.open(package_path, "r:gz") as archive:
                    _safe_extract(archive, extracted)
            pdfs = sorted(extracted.rglob("*.pdf"))
            if not pdfs:
                raise ValueError("OA package has no PDF")
            article_pdf = max(pdfs, key=lambda path: path.stat().st_size)
            destination = fulltext / f"{record.source_id}_{pmcid}.pdf"
            if not destination.exists():
                shutil.copy2(article_pdf, destination)
            support_dir = supporting / record.source_id
            support_dir.mkdir(exist_ok=True)
            supplementary_files = [
                path for path in extracted.rglob("*") if path.is_file() and path != article_pdf
            ]
            rows.append(
                {
                    **base,
                    "status": "oa_package_verified",
                    "license": record_node.attrib.get("license", ""),
                    "retracted": record_node.attrib.get("retracted", ""),
                    "package_url": package_url,
                    "package_path": str(package_path),
                    "package_bytes": package_path.stat().st_size,
                    "package_sha256": _sha256(package_path),
                    "article_pdf": str(destination),
                    "article_pdf_bytes": destination.stat().st_size,
                    "article_pdf_sha256": _sha256(destination),
                    "article_pdf_pages": _pdf_pages(destination),
                    "package_nonarticle_files": len(supplementary_files),
                }
            )
        except Exception as exc:
            try:
                rows.append(
                    {
                        **base,
                        **_europe_pmc_xml(record.source_id, pmcid, fulltext_xml),
                        "oa_package_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except Exception as fallback_exc:
                rows.append(
                    {
                        **base,
                        "status": "download_or_validation_error",
                        "error": (
                            f"OA package {type(exc).__name__}: {exc}; "
                            f"XML fallback {type(fallback_exc).__name__}: {fallback_exc}"
                        ),
                    }
                )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(project / "provenance" / "literature_download_manifest.csv", index=False)
    summary = {
        "records": int(len(manifest)),
        "status_counts": manifest["status"].value_counts().to_dict(),
        "rule": "No paywall bypass; full OA packages retained when supplied by PMC.",
    }
    (project / "provenance" / "literature_download_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
