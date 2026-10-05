"""Check manuscript statistics, citation keys, and submission-level text invariants."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MANUSCRIPT = PROJECT / "manuscript"
BASELINE = PROJECT / "provenance" / "manuscript_text_invariants.json"

SOURCE_FILES = [
    MANUSCRIPT / "chapters" / "01_introduction_framework.md",
    MANUSCRIPT / "chapters" / "02_methods.md",
    MANUSCRIPT / "chapters" / "03_results.md",
    MANUSCRIPT / "chapters" / "04_discussion.md",
    MANUSCRIPT / "front_back_matter.md",
    MANUSCRIPT / "supplement.md",
    MANUSCRIPT / "tables.md",
    MANUSCRIPT / "figure_legends.md",
    MANUSCRIPT / "cover_letter_brm.md",
]

NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:[−-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+−-]?\d+)?%?)(?![A-Za-z])", re.I)
KEY_RE = re.compile(r"\b(?:LIT|MET)\d{3}\b")


def source_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in SOURCE_FILES)


def counter_payload(pattern: re.Pattern[str], text: str) -> dict[str, int]:
    return dict(sorted(Counter(pattern.findall(text)).items()))


def freeze() -> None:
    text = source_text()
    payload = {
        "scope": [str(path.relative_to(PROJECT)) for path in SOURCE_FILES],
        "number_tokens": counter_payload(NUMBER_RE, text),
        "citation_keys": counter_payload(KEY_RE, text),
        "note": "Created after D10 disclosure and before prose-only humanization passes.",
    }
    BASELINE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"frozen={BASELINE}")


def check_baseline() -> list[str]:
    failures: list[str] = []
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    text = source_text()
    current_numbers = counter_payload(NUMBER_RE, text)
    current_keys = counter_payload(KEY_RE, text)
    if current_numbers != expected["number_tokens"]:
        failures.append("source numeric-token multiset changed after de-AI freeze")
    if current_keys != expected["citation_keys"]:
        failures.append("source citation-key multiset changed after de-AI freeze")
    return failures


def check_assembled() -> list[str]:
    failures: list[str] = []
    text = (MANUSCRIPT / "manuscript_anonymized.md").read_text(encoding="utf-8")
    if KEY_RE.search(text):
        failures.append("assembled manuscript contains unconverted LIT/MET key")

    abstract_match = re.search(r"## Abstract\n\n(.*?)\n\n\*\*Keywords", text, flags=re.S)
    if not abstract_match:
        failures.append("abstract block not found")
    else:
        abstract_words = len(re.findall(r"\b[\w'-]+\b", abstract_match.group(1)))
        if abstract_words > 250:
            failures.append(f"abstract has {abstract_words} words (>250)")

    required_fragments = [
        "251 complete pairs, 502 records, and 52 participants",
        "104 unique participants: 30 Short-R, 25 Long, 26 Schema, and 23 Short-P",
        "ROC AUC was .877",
        "95% CI [.817, .932]",
        "contrast was −.7539",
        "was −.04284",
        "*p* = .00130",
        "*p* = .52255",
        "72 (60%)",
        "coefficient for held-out EEG CorrCA ISC was .01784",
        "*p* = .024",
        "coefficient was 0.341 for remember",
        "2.441 for detail",
        "without boundary-timing specificity",
    ]
    for fragment in required_fragments:
        if fragment not in text:
            failures.append(f"required manuscript fragment missing: {fragment}")

    prohibited = [
        r"\beye(?:-tracking| tracking| movement| synchrony| ISC)? (?:proves?|demonstrates?) (?:attention|event segmentation)\b",
        r"\b(?:demonstrates?|proves?|establishes?) (?:dopamine release|pleasure|brain damage)\b",
        r"\b(?:a|the) biomarker of attention\b",
    ]
    for pattern in prohibited:
        if re.search(pattern, text, flags=re.I):
            failures.append(f"prohibited claim pattern found: {pattern}")

    dois: list[str] = []
    current_record: list[str] = []
    for line in (MANUSCRIPT / "references.ris").read_text(encoding="utf-8").splitlines():
        if line.startswith("DO  - "):
            dois.append(line[6:].strip().lower())
        if line.startswith("TY  - "):
            current_record = [line]
        elif current_record:
            current_record.append(line)
            if line == "ER  -":
                if not any(item.startswith("AU  - ") for item in current_record):
                    failures.append("RIS record without author")
                current_record = []
    if len(dois) != len(set(dois)):
        failures.append("duplicate DOI in RIS")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return 0
    failures = check_baseline() + check_assembled()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("manuscript_audit=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
