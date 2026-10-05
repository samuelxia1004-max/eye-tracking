# Data not redistributed in this code_release

Raw BBBD / Sherlock downloads and large derived EEG arrays under `data/raw/` and `data/derived/bbbd_eeg_16hz/` (tens of GB locally) are **not** included.

To reproduce:
1. Download BBBD from Zenodo (Madsen et al.; DOI 10.5281/zenodo.19241964 — verify the current record) under the dataset terms.
2. Obtain Sherlock eye-tracking / boundary materials from the OSF project cited in the manuscript (do not redistribute commercial video).
3. Point paths in `config/analysis.json` to your local copies, then follow README.md.

Small processed arrays (`*.npz` at `data/derived/` root, if present) may be included for convenience; regenerate from raw if hashes differ.

Note (2026-10-06 GitHub release): the convenience `*.npz` caches under `data/derived/` were omitted from the public repository (regenerate locally from public BBBD/Sherlock downloads; see README).
