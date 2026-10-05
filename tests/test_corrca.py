import numpy as np

from boundaryisc.corrca import fit_corrca, held_participant_corrca_isc, project_corrca


def test_corrca_recovers_shared_component_and_held_isc():
    rng = np.random.default_rng(41)
    timepoints = 800
    shared = rng.normal(size=timepoints)
    mixing = np.array([1.0, -0.7, 0.4, 0.1])
    participants = np.stack(
        [shared[:, None] * mixing + rng.normal(scale=0.5, size=(timepoints, 4)) for _ in range(10)]
    )
    fit = fit_corrca(participants[:8], components=2)
    projected = project_corrca(participants[:8], fit)
    assert projected.shape == (8, timepoints, 2)
    score, correlations, _ = held_participant_corrca_isc(
        participants[:8], participants[8], components=2
    )
    assert correlations[0] > 0.7
    assert np.isfinite(score)
