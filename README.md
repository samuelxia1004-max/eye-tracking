# Crossed-null naturalistic eye-tracking analysis (code release)

Companion code for the preprint manuscript:

> **Crossed randomization distinguishes global ocular-score differences from event-boundary-specific effects in naturalistic eye tracking**

Author for public release: **Samuel Xia** (fill affiliation / ORCID on the preprint title page).

License: **MIT** (see `LICENSE`).

## What this package does

- Trains an ocular distracted-session benchmark on public **BBBD** (Brain, Body, and Behavior Dataset) eye-tracking sessions.
- Freezes the model and transports it to public **Sherlock** commercial-film eye-tracking records.
- Separates a **participant-label** null from a **boundary-location** (temporal) null via an intersection-union (IUT) decision.
- Runs a frozen specification grid and AR(1)-style simulations (see `analysis_plan.md` and `config/`).

This folder was assembled from the local project
`~/Documents/Codex/2026-08-14/i/outputs/short-video-dual-null-attention`
(with absolute paths sanitized and raw multi-GB archives omitted).

## What is NOT included

- Raw BBBD ZIP archives and large derived EEG arrays (`data/raw/`, `data/derived/bbbd_eeg_16hz/`, ~tens of GB).
- Commercial Sherlock video (must not be redistributed).
- See `DATA_NOT_INCLUDED.md`.

Small processed arrays may be present under `data/derived/*.npz` for convenience; regenerate from raw sources if needed.

## Dataset download links / citations (verify before posting)

1. **BBBD** — Madsen, Kuppa, Parra et al., Brain–Body–Behavior Dataset. Zenodo record cited in the manuscript / DataCite DOI **10.5281/zenodo.19241964** (confirm the current version on Zenodo before citing). Follow the dataset’s own license/terms.
2. **Sherlock eye-tracking / event boundaries** — use the OSF / deposited packages cited in the manuscript References (e.g., OSF Li event materials and related Sherlock segmentations). Do **not** upload the commercial film to GitHub.
3. Point the placeholder paths in `config/analysis.json` to your lawful local copies. The original frozen config also referenced processed Sherlock caches from the sibling local project `event-boundary-isc-robustness` (`commercial_10hz.npz`, participant manifest); recreate those via the package pipeline or copy from your local archive **without** committing raw video.

## How to reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements-lock.txt   # or: pip install -r requirements.txt
pip install -e .

# Edit config/analysis.json placeholders to local paths first
PYTHONPATH=src python -m boundaryisc.cli --config config/analysis.json validate-config
PYTHONPATH=src python -m boundaryisc.cli --config config/analysis.json reproduce
PYTHONPATH=src python -m boundaryisc.cli --config config/analysis.json analyze
PYTHONPATH=src python -m boundaryisc.cli --config config/analysis.json simulate
PYTHONPATH=src python -m boundaryisc.cli --config config/analysis.json report
pytest -q
```

Frozen seed and expected counts are in `config/analysis.json` and `analysis_plan.md`. Full numerical re-runs need the raw archives; table outputs under `results/` document the completed local run.

## Reproducibility status (honest)

- **Code found:** yes — Python package `boundaryisc` under `src/`, CLI, tests, frozen configs, result tables.
- **Fully reproducible from this folder alone:** **no** — raw BBBD/Sherlock inputs and some Sherlock cache files are external; `config/analysis.json` must be pointed at local lawful copies.
- **Heavy downloads:** not performed by this packaging step (by design).

## Related local sibling (not copied here)

An earlier Sherlock-focused package lives at
`~/Documents/Codex/2026-08-14/i/outputs/event-boundary-isc-robustness`
and supplied some Sherlock derived caches referenced by the dual-null config. Consider merging those derived files into your private data directory when reproducing, or regenerating them.

## Pushing to GitHub

1. Create a new empty public repository.
2. Copy this `code_release/` folder contents to the repo root (do not commit raw data).
3. `git init && git add . && git commit -m "Initial public analysis package" && git remote add origin <url> && git push -u origin main`
4. Paste the repo URL into the preprint Data/Code availability placeholders.
