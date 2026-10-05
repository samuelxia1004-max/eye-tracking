"""Read BBBD ocular derivatives directly from official experiment archives."""

from __future__ import annotations

import gzip
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


_SIGNAL_PATTERN = re.compile(
    r"(?:^|/)derivatives/sub-(?P<subject>\d+)/ses-(?P<session>\d+)/eyetrack/"
    r"sub-\d+_ses-\d+_task-stim(?P<stimulus>\d+)_desc-(?P<modality>gaze|pupil)"
    r"(?P<visualangle>_visualangle)?_eyetrack\.tsv\.gz$"
)
_INTERPOLATION_PATTERN = re.compile(
    r"(?:^|/)derivatives/sub-(?P<subject>\d+)/ses-(?P<session>\d+)/eyetrack/"
    r"sub-\d+_ses-\d+_task-stim(?P<stimulus>\d+)_desc-(?P<modality>gaze|pupil)"
    r"_interpolation_timestamps\.tsv\.gz$"
)


@dataclass(frozen=True)
class BBBDOcularCache:
    data: np.ndarray
    original_valid_fraction: np.ndarray
    participant: np.ndarray
    video: np.ndarray
    condition: np.ndarray
    experiment: np.ndarray
    session: np.ndarray
    original_length: np.ndarray


def _read_numeric_gzip(payload: bytes) -> np.ndarray:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
        frame = pd.read_csv(stream, sep="\t", header=None)
    return frame.to_numpy(dtype=float)


def _read_intervals_gzip(payload: bytes) -> np.ndarray:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
        frame = pd.read_csv(stream, sep="\t")
    required = {"start_time", "end_time"}
    if not required.issubset(frame.columns):
        raise ValueError(f"interpolation file lacks {sorted(required)}")
    intervals = frame.loc[:, ["start_time", "end_time"]].to_numpy(dtype=float)
    if intervals.size and (
        not np.isfinite(intervals).all() or np.any(intervals[:, 1] < intervals[:, 0])
    ):
        raise ValueError("invalid interpolation intervals")
    return intervals


def scan_bbbd_ocular_archive(path: str | Path, experiment: int) -> pd.DataFrame:
    """Return one manifest row per participant-session-video record."""

    records: dict[tuple[int, int, int], dict[str, object]] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            signal = _SIGNAL_PATTERN.search(info.filename)
            interpolation = _INTERPOLATION_PATTERN.search(info.filename)
            match = signal or interpolation
            if match is None:
                continue
            key = (
                int(match.group("subject")),
                int(match.group("session")),
                int(match.group("stimulus")),
            )
            row = records.setdefault(
                key,
                {
                    "experiment": int(experiment),
                    "subject": key[0],
                    "session": key[1],
                    "stimulus": key[2],
                },
            )
            modality = match.group("modality")
            if interpolation is not None:
                row[f"{modality}_interpolation_member"] = info.filename
                row[f"{modality}_interpolation_compressed_bytes"] = int(info.compress_size)
            else:
                row[f"{modality}_member"] = info.filename
                row[f"{modality}_compressed_bytes"] = int(info.compress_size)
                row[f"{modality}_uncompressed_bytes"] = int(info.file_size)
    frame = pd.DataFrame(records.values())
    if frame.empty:
        raise ValueError(f"no BBBD ocular derivatives found in {path}")
    for column in (
        "gaze_member",
        "pupil_member",
        "gaze_interpolation_member",
        "pupil_interpolation_member",
    ):
        if column not in frame:
            frame[column] = np.nan
    frame["has_gaze"] = frame["gaze_member"].notna()
    frame["has_pupil"] = frame["pupil_member"].notna()
    frame["has_gaze_interpolation"] = frame["gaze_interpolation_member"].notna()
    frame["has_pupil_interpolation"] = frame["pupil_interpolation_member"].notna()
    frame["complete_modalities"] = frame["has_gaze"] & frame["has_pupil"]
    frame["complete_validity_masks"] = (
        frame["has_gaze_interpolation"] & frame["has_pupil_interpolation"]
    )
    frame["participant"] = frame.apply(
        lambda row: f"exp{int(row.experiment)}_sub-{int(row.subject):02d}", axis=1
    )
    frame["video"] = frame.apply(
        lambda row: f"exp{int(row.experiment)}_stim-{int(row.stimulus):02d}", axis=1
    )
    frame["condition"] = frame["session"].map({1: "attentive", 2: "distracted"})
    return frame.sort_values(["subject", "stimulus", "session"]).reset_index(drop=True)


def _downsample_128_to_10(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("released processed ocular derivative contains non-finite samples")
    return resample_poly(values, up=5, down=64, axis=0, padtype="line")


def _validity_at_10hz(length: int, intervals: np.ndarray, output_length: int) -> np.ndarray:
    valid = np.ones(length, dtype=float)
    for start_seconds, end_seconds in intervals:
        start = max(0, int(np.floor(start_seconds * 128.0)))
        stop = min(length, int(np.ceil(end_seconds * 128.0)))
        valid[start:stop] = 0.0
    edges = np.linspace(0, length, output_length + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(valid)])
    fractions = np.empty(output_length, dtype=float)
    for index in range(output_length):
        start = int(np.floor(edges[index]))
        stop = min(length, max(int(np.ceil(edges[index + 1])), start + 1))
        fractions[index] = (cumulative[stop] - cumulative[start]) / (stop - start)
    return np.clip(fractions, 0.0, 1.0)


def load_bbbd_ocular_archives(
    archives: Iterable[tuple[int, str | Path]],
    *,
    require_paired_sessions: bool = True,
) -> tuple[BBBDOcularCache, pd.DataFrame]:
    """Load X, Y, and pupil at 10 Hz and pad records with NaN."""

    arrays: list[np.ndarray] = []
    validities: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    audit_frames: list[pd.DataFrame] = []
    for experiment, path in archives:
        manifest = scan_bbbd_ocular_archive(path, experiment)
        audit_frames.append(manifest)
        eligible = manifest.loc[
            manifest["complete_modalities"]
            & manifest["complete_validity_masks"]
            & manifest["condition"].notna()
        ].copy()
        if require_paired_sessions:
            session_count = eligible.groupby(["participant", "video"])["condition"].nunique()
            paired = session_count[session_count == 2].index
            eligible = eligible.set_index(["participant", "video"]).loc[paired].reset_index()
        with zipfile.ZipFile(path) as archive:
            for row in eligible.itertuples(index=False):
                gaze = _read_numeric_gzip(archive.read(row.gaze_member))
                pupil = _read_numeric_gzip(archive.read(row.pupil_member))
                if gaze.ndim != 2 or gaze.shape[1] < 2:
                    raise ValueError(f"gaze file has invalid shape: {row.gaze_member} {gaze.shape}")
                if pupil.ndim != 2 or pupil.shape[1] < 1:
                    raise ValueError(f"pupil file has invalid shape: {row.pupil_member} {pupil.shape}")
                if gaze.shape[0] != pupil.shape[0]:
                    raise ValueError(f"gaze/pupil length mismatch for {row.participant} {row.video}")
                if gaze.shape[1] < 4:
                    raise ValueError(
                        f"released gaze visual-angle file needs four columns: {row.gaze_member}"
                    )
                ocular = np.column_stack([gaze[:, -2], gaze[:, -1], pupil[:, 0]])
                downsampled = _downsample_128_to_10(ocular).astype(np.float32)
                gaze_intervals = _read_intervals_gzip(
                    archive.read(row.gaze_interpolation_member)
                )
                pupil_intervals = _read_intervals_gzip(
                    archive.read(row.pupil_interpolation_member)
                )
                gaze_validity = _validity_at_10hz(
                    ocular.shape[0], gaze_intervals, downsampled.shape[0]
                )
                pupil_validity = _validity_at_10hz(
                    ocular.shape[0], pupil_intervals, downsampled.shape[0]
                )
                validity = np.column_stack(
                    [gaze_validity, gaze_validity, pupil_validity]
                ).astype(np.float32)
                arrays.append(downsampled)
                validities.append(validity)
                metadata.append(
                    {
                        "participant": row.participant,
                        "video": row.video,
                        "condition": row.condition,
                        "experiment": int(row.experiment),
                        "session": int(row.session),
                        "original_length": int(ocular.shape[0]),
                        "downsampled_length": int(downsampled.shape[0]),
                        "gaze_member": row.gaze_member,
                        "pupil_member": row.pupil_member,
                        "gaze_interpolation_member": row.gaze_interpolation_member,
                        "pupil_interpolation_member": row.pupil_interpolation_member,
                        "gaze_original_valid_fraction": float(gaze_validity.mean()),
                        "pupil_original_valid_fraction": float(pupil_validity.mean()),
                    }
                )
    if not arrays:
        raise ValueError("no paired BBBD ocular records were loadable")
    max_length = max(array.shape[0] for array in arrays)
    data = np.full((len(arrays), max_length, 3), np.nan, dtype=np.float32)
    original_valid_fraction = np.full_like(data, np.nan)
    for index, (array, validity) in enumerate(zip(arrays, validities, strict=True)):
        data[index, : array.shape[0]] = array
        original_valid_fraction[index, : validity.shape[0]] = validity
    meta = pd.DataFrame(metadata)
    cache = BBBDOcularCache(
        data=data,
        original_valid_fraction=original_valid_fraction,
        participant=meta["participant"].to_numpy(str),
        video=meta["video"].to_numpy(str),
        condition=meta["condition"].to_numpy(str),
        experiment=meta["experiment"].to_numpy(int),
        session=meta["session"].to_numpy(int),
        original_length=meta["original_length"].to_numpy(int),
    )
    audit = pd.concat(audit_frames, ignore_index=True)
    return cache, audit


def save_bbbd_ocular_cache(cache: BBBDOcularCache, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        data=cache.data,
        original_valid_fraction=cache.original_valid_fraction,
        participant=cache.participant,
        video=cache.video,
        condition=cache.condition,
        experiment=cache.experiment,
        session=cache.session,
        original_length=cache.original_length,
        channels=np.asarray(["x", "y", "pupil"]),
        rate_hz=np.asarray(10),
    )
