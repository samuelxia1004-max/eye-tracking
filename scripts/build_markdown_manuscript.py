"""Assemble the anonymous APA-style Markdown manuscript from reviewed sources."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MANUSCRIPT = PROJECT / "manuscript"

CITATIONS = {
    "LIT001": ("berger", "Berger & Hsu, 1996"),
    "LIT002": ("chen", "Chen et al., 2016"),
    "LIT003": ("nastase", "Nastase et al., 2019"),
    "LIT004": ("ben-yakov", "Ben-Yakov & Henson, 2018"),
    "LIT005": ("silva", "Silva et al., 2019"),
    "LIT006": ("yates", "Yates et al., 2022"),
    "LIT007": ("sava-segal", "Sava-Segal et al., 2023"),
    "LIT008": ("benear", "Benear et al., 2023"),
    "LIT009": ("sasmita", "Sasmita & Swallow, 2023"),
    "LIT010": ("al-zahli", "Al-Zahli et al., 2026"),
    "LIT011": ("baror", "Baror et al., 2026"),
    "LIT012": ("madsen-2021", "Madsen et al., 2021"),
    "LIT013": ("madsen-parra-2022", "Madsen & Parra, 2022"),
    "LIT014": ("madsen-parra-2025", "Madsen & Parra, 2025"),
    "LIT015": ("madsen-kuppa-2026a", "Madsen et al., 2026a"),
    "LIT016": ("madsen-kuppa-2026b", "Madsen et al., 2026b"),
    "LIT017": ("li-j-2025", "J. Li et al., 2025"),
    "LIT018": ("li-h-2025", "H. Li et al., 2025"),
    "LIT019": ("hasson", "Hasson et al., 2004"),
    "MET001": ("owen", "Owen & Eckles, 2012"),
    "MET002": ("phipson", "Phipson & Smyth, 2010"),
    "MET003": ("yarkoni", "Yarkoni, 2022"),
    "MET004": ("varma", "Varma & Simon, 2006"),
    "MET005": ("varoquaux", "Varoquaux, 2018"),
    "MET006": ("winkler", "Winkler et al., 2014"),
    "MET007": ("westfall", "Westfall et al., 2014"),
    "MET008": ("baayen", "Baayen et al., 2008"),
}

APA_SORT = {
    key: index
    for index, key in enumerate(
        [
            "al-zahli", "baayen", "baror", "benear", "ben-yakov", "berger",
            "chen", "hasson", "li-h-2025", "li-j-2025", "madsen-2021",
            "madsen-kuppa-2026a", "madsen-kuppa-2026b", "madsen-parra-2022",
            "madsen-parra-2025", "nastase", "owen", "phipson", "sasmita",
            "sava-segal", "silva", "varma", "varoquaux", "westfall", "winkler",
            "yarkoni", "yates",
        ]
    )
}


def convert_citations(text: str) -> str:
    pattern = re.compile(r"\[((?:LIT|MET)\d{3}(?:,\s*(?:LIT|MET)\d{3})*)\]")

    def replacement(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",")]
        ordered = sorted(keys, key=lambda key: APA_SORT[CITATIONS[key][0]])
        return "(" + "; ".join(CITATIONS[key][1] for key in ordered) + ")"

    text = pattern.sub(replacement, text)
    return text.replace("Madsen et al. (2026)", "Madsen et al. (2026a, 2026b)")


def figures_block() -> str:
    legends = (MANUSCRIPT / "figure_legends.md").read_text(encoding="utf-8")
    entries = re.findall(
        r"\*\*Figure (\d+)\. ([^*]+)\.\*\*\s*(.*?)(?=\n\n\*\*Figure|\Z)",
        legends,
        flags=re.S,
    )
    image_names = {
        "1": "figure1_framework.png",
        "2": "figure2_bbbd_benchmark.png",
        "3": "figure3_sherlock_crossed_null.png",
        "4": "figure4_simulation.png",
    }
    blocks = ["# Figures"]
    for number, title, body in entries:
        blocks.extend(
            [
                f"## Figure {number}. {title}",
                f"![Figure {number}. {title}](../results/figures/{image_names[number]})",
                body.strip(),
            ]
        )
    return "\n\n".join(blocks)


def build() -> str:
    front_back = (MANUSCRIPT / "front_back_matter.md").read_text(encoding="utf-8")
    front, back = front_back.split("## Acknowledgements", maxsplit=1)
    back = "## Acknowledgements" + back
    chapters = [
        (MANUSCRIPT / "chapters" / name).read_text(encoding="utf-8")
        for name in (
            "01_introduction_framework.md",
            "02_methods.md",
            "03_results.md",
            "04_discussion.md",
        )
    ]
    references = (MANUSCRIPT / "references_apa.md").read_text(encoding="utf-8")
    tables = (MANUSCRIPT / "tables.md").read_text(encoding="utf-8")
    text = "\n\n".join([front.strip(), *[c.strip() for c in chapters], back.strip(), references.strip(), tables.strip(), figures_block()])
    text = convert_citations(text)
    if re.search(r"\[(?:LIT|MET)\d{3}", text):
        raise RuntimeError("unconverted source-key citation")
    return text.strip() + "\n"


if __name__ == "__main__":
    assembled = build()
    for filename in ("manuscript.md", "manuscript_anonymized.md"):
        (MANUSCRIPT / filename).write_text(assembled, encoding="utf-8")
    print(f"assembled_words={len(assembled.split())}")
