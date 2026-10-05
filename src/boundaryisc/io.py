"""Configuration, provenance, and public OSF input handling."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat


EXPECTED_GROUP_SIZES = {"Short-R": 30, "Long": 25, "Schema": 26, "Short-P": 23}


def load_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config, config_path


def resolve_project_path(config: dict, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["project_root"]) / path


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_manifest(rows: list[dict[str, str]], expected: dict[str, int] | None = None) -> None:
    expected = expected or EXPECTED_GROUP_SIZES
    ids = [int(row["participant_id"]) for row in rows]
    if len(rows) != 104 or len(set(ids)) != 104:
        raise ValueError(f"expected 104 unique commercial participants, found {len(set(ids))}")
    counts = Counter(row["group"] for row in rows)
    if dict(counts) != expected:
        raise ValueError(f"group counts mismatch: {dict(counts)} != {expected}")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_commercial_cache(config: dict, *, force: bool = False) -> Path:
    manifest_path = resolve_project_path(config, config["participant_manifest"])
    output_path = resolve_project_path(config, config["derived_cache"])
    if output_path.exists() and not force:
        return output_path
    rows = read_manifest(manifest_path)
    validate_manifest(rows, config["expected_group_sizes"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    expected_samples = int(config["expected_samples"])
    data = np.empty((n, expected_samples, 3), dtype=np.float32)
    data_zip = Path(config["inputs"]["data_zip"])
    variables = config["mat_variables"]
    step = int(config["downsample_step"])
    with zipfile.ZipFile(data_zip) as archive:
        for i, row in enumerate(rows):
            payload = archive.read(row["mat_member"])
            mat = loadmat(io.BytesIO(payload), variable_names=variables, squeeze_me=True)
            for channel_index, variable in enumerate(variables):
                if variable not in mat:
                    raise ValueError(f"{variable} missing for participant {row['participant_id']}")
                series = np.asarray(mat[variable], dtype=float).reshape(-1)
                reduced = series[::step]
                if reduced.size != expected_samples:
                    raise ValueError(
                        f"participant {row['participant_id']} has {reduced.size} samples; expected {expected_samples}"
                    )
                if not np.isfinite(reduced).all():
                    raise ValueError(f"non-finite samples for participant {row['participant_id']}")
                data[i, :, channel_index] = reduced.astype(np.float32)
    np.savez(
        output_path,
        data=data,
        participant_id=np.asarray([int(row["participant_id"]) for row in rows]),
        group=np.asarray([row["group"] for row in rows]),
        remember=np.asarray([float(row["remember"]) for row in rows]),
        detail=np.asarray([float(row["detail"]) for row in rows]),
        channel=np.asarray(config["channels"]),
    )
    return output_path


def load_commercial_cache(config: dict) -> dict[str, np.ndarray]:
    path = prepare_commercial_cache(config)
    loaded = np.load(path, allow_pickle=False)
    result = {key: loaded[key] for key in loaded.files}
    if result["data"].shape != (104, config["expected_samples"], 3):
        raise ValueError(f"cache shape mismatch: {result['data'].shape}")
    if len(config["boundaries_10hz"][: config["usable_boundary_count"]]) != 18:
        raise ValueError("expected 18 usable boundaries")
    return result


def participant_id_from_member(member: str) -> int | None:
    match = re.search(r"Test(\d+)", Path(member).name)
    return int(match.group(1)) if match else None
