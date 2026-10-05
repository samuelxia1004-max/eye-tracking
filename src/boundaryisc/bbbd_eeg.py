"""BBBD EEG archive scanning, 16-Hz caching, and held-participant CorrCA."""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .attention import double_block_splits
from .corrca import held_participant_corrca_isc


_EEG_PATTERN = re.compile(
    r"(?:^|/)derivatives/sub-(?P<subject>\d+)/ses-(?P<session>\d+)/eeg/"
    r"sub-\d+_ses-\d+_task-stim(?P<stimulus>\d+)_desc-eeg\.bdf$"
)


def scan_bbbd_eeg_archive(path: str | Path, experiment: int) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            match = _EEG_PATTERN.search(info.filename)
            if match is None:
                continue
            subject = int(match.group("subject"))
            session = int(match.group("session"))
            stimulus = int(match.group("stimulus"))
            rows.append(
                {
                    "experiment": int(experiment),
                    "subject": subject,
                    "session": session,
                    "stimulus": stimulus,
                    "participant": f"exp{experiment}_sub-{subject:02d}",
                    "video": f"exp{experiment}_stim-{stimulus:02d}",
                    "condition": {1: "attentive", 2: "distracted"}.get(session),
                    "member": info.filename,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no derived EEG BDF members in {path}")
    return frame.sort_values(["subject", "stimulus", "session"]).reset_index(drop=True)


def cache_bbbd_eeg(
    archives: list[tuple[int, str | Path]],
    output_dir: str | Path,
    *,
    output_rate_hz: int = 16,
) -> pd.DataFrame:
    """Read each BDF member through MNE and save a compact float32 array."""

    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to read BDF EEG") from exc
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for experiment, archive_path in archives:
        manifest = scan_bbbd_eeg_archive(archive_path, experiment)
        with zipfile.ZipFile(archive_path) as archive:
            for row in manifest.itertuples(index=False):
                filename = (
                    f"{row.participant}_{row.video}_{row.condition}_eeg_{output_rate_hz}hz.npy"
                )
                target = destination / filename
                if not target.exists():
                    payload = archive.read(row.member)
                    with tempfile.NamedTemporaryFile(suffix=".bdf") as temporary:
                        temporary.write(payload)
                        temporary.flush()
                        raw = mne.io.read_raw_bdf(temporary.name, preload=True, verbose="ERROR")
                    source_rate = int(round(raw.info["sfreq"]))
                    eeg = raw.get_data()[:64].T
                    reduced = resample_poly(
                        eeg, up=output_rate_hz, down=source_rate, axis=0, padtype="line"
                    )
                    reduced = reduced - reduced.mean(axis=0, keepdims=True)
                    np.save(target, reduced.astype(np.float32))
                array = np.load(target, mmap_mode="r")
                output_rows.append(
                    {
                        **row._asdict(),
                        "cache_path": str(target.resolve()),
                        "samples": int(array.shape[0]),
                        "channels": int(array.shape[1]),
                        "rate_hz": int(output_rate_hz),
                    }
                )
    return pd.DataFrame(output_rows)


def cross_validated_corrca_scores(
    manifest: pd.DataFrame,
    *,
    participant_folds: int = 5,
    components: int = 3,
    seed: int = 20260820,
) -> pd.DataFrame:
    """Fit per-video/session CorrCA filters without each held participant fold."""

    participants = manifest["participant"].to_numpy(object)
    videos = manifest["video"].to_numpy(object)
    fold_by_participant: dict[str, int] = {}
    for _, test, fold, _ in double_block_splits(
        participants, videos, participant_folds=participant_folds, seed=seed
    ):
        for participant in np.unique(participants[test]):
            fold_by_participant[str(participant)] = int(fold)
    rows = []
    for (video, condition), block in manifest.groupby(["video", "condition"], sort=False):
        block = block.reset_index(drop=True)
        for fold in range(participant_folds):
            held_mask = block["participant"].map(fold_by_participant).to_numpy(int) == fold
            train_block = block.loc[~held_mask]
            held_block = block.loc[held_mask]
            if held_block.empty or len(train_block) < 3:
                continue
            arrays = [np.load(path) for path in train_block["cache_path"]]
            held_arrays = [np.load(path) for path in held_block["cache_path"]]
            common_length = min([array.shape[0] for array in arrays + held_arrays])
            training = np.stack([array[:common_length] for array in arrays]).astype(float)
            for held_row, held_array in zip(held_block.itertuples(index=False), held_arrays, strict=True):
                score, component_r, fit = held_participant_corrca_isc(
                    training,
                    held_array[:common_length].astype(float),
                    components=components,
                )
                rows.append(
                    {
                        "participant": held_row.participant,
                        "video": video,
                        "condition": condition,
                        "participant_fold": fold,
                        "eeg_corrca_isc": score,
                        "component_r": json.dumps(component_r.tolist()),
                        "component_eigenvalues": json.dumps(fit.eigenvalues.tolist()),
                        "training_participants": len(train_block),
                        "samples": common_length,
                        "components": components,
                    }
                )
    result = pd.DataFrame(rows)
    expected = manifest[["participant", "video", "condition"]].drop_duplicates()
    merged = expected.merge(result, on=["participant", "video", "condition"], how="left")
    if merged["eeg_corrca_isc"].isna().any():
        missing = merged.loc[merged["eeg_corrca_isc"].isna(), ["participant", "video", "condition"]]
        raise ValueError(f"missing held EEG scores for {len(missing)} records")
    return merged
