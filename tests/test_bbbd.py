import gzip
import io
import zipfile

import numpy as np

from boundaryisc.bbbd import load_bbbd_ocular_archives, scan_bbbd_ocular_archive


def _tsv_gzip(values):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as stream:
        np.savetxt(stream, values, delimiter="\t")
    return buffer.getvalue()


def _interval_gzip(intervals):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as stream:
        stream.write(b"start_time\tend_time\n")
        np.savetxt(stream, intervals, delimiter="\t")
    return buffer.getvalue()


def test_scan_and_load_synthetic_bbbd_archive(tmp_path):
    archive_path = tmp_path / "experiment2.zip"
    time = np.linspace(0, 4 * np.pi, 2560)
    with zipfile.ZipFile(archive_path, "w") as archive:
        for subject in (1, 2):
            for session in (1, 2):
                base = f"experiment2/derivatives/sub-{subject:02d}/ses-{session:02d}/eyetrack"
                prefix = f"sub-{subject:02d}_ses-{session:02d}_task-stim01"
                gaze = np.column_stack(
                    [np.zeros((time.size, 2)), np.sin(time), np.cos(time)]
                )
                pupil = (2 + 0.2 * np.sin(time))[:, None]
                archive.writestr(
                    f"{base}/{prefix}_desc-gaze_visualangle_eyetrack.tsv.gz",
                    _tsv_gzip(gaze),
                )
                archive.writestr(f"{base}/{prefix}_desc-pupil_eyetrack.tsv.gz", _tsv_gzip(pupil))
                archive.writestr(
                    f"{base}/{prefix}_desc-gaze_interpolation_timestamps.tsv.gz",
                    _interval_gzip(np.array([[1.0, 1.2]])),
                )
                archive.writestr(
                    f"{base}/{prefix}_desc-pupil_interpolation_timestamps.tsv.gz",
                    _interval_gzip(np.array([[2.0, 2.1]])),
                )
    manifest = scan_bbbd_ocular_archive(archive_path, 2)
    assert len(manifest) == 4
    assert manifest["complete_modalities"].all()
    cache, audit = load_bbbd_ocular_archives([(2, archive_path)])
    assert cache.data.shape == (4, 200, 3)
    assert cache.original_valid_fraction.shape == cache.data.shape
    assert np.nanmin(cache.original_valid_fraction) < 1.0
    assert set(cache.condition) == {"attentive", "distracted"}
    assert len(audit) == 4
