import numpy as np

from boundaryisc import boundary_contrast, compute_isc, simulate_recovery


def test_perfect_synchrony_is_one():
    series = np.linspace(-1, 1, 100)
    data = np.stack([series, series, series])[:, :, None]
    scores = compute_isc(data, ["g", "g", "g"], channel="x")
    assert np.allclose(scores, 1.0)


def test_independent_noise_is_near_zero():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(40, 2000, 1))
    scores = compute_isc(data, ["g"] * 40, channel="x")
    assert abs(np.nanmean(scores)) < 0.03


def test_known_boundary_effect_has_expected_direction():
    rng = np.random.default_rng(9)
    pre = rng.normal(size=(20, 200, 1))
    boundary = rng.normal(size=(20, 200, 1))
    shared = rng.normal(size=200)
    boundary[10:, :, 0] += 1.2 * shared
    groups = np.asarray(["target"] * 10 + ["control"] * 10)
    pre_scores = compute_isc(pre, groups, channel="x")
    boundary_scores = compute_isc(boundary, groups, channel="x")
    result = boundary_contrast(boundary_scores, pre_scores, groups, target_group="target")
    assert result.effect < -0.2


def test_single_channel_effect_is_diluted_in_composite():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(12, 300, 3))
    shared = rng.normal(size=300)
    data[:, :, 0] += shared
    groups = ["g"] * 12
    x_score = np.nanmean(compute_isc(data, groups, channel="x"))
    composite = np.nanmean(compute_isc(data, groups, channel="composite"))
    assert x_score > composite


def test_constant_window_returns_nan():
    data = np.ones((3, 40, 1))
    scores = compute_isc(data, ["g", "g", "g"], channel="x")
    assert np.isnan(scores).all()


def test_missing_partner_returns_nan():
    data = np.arange(30, dtype=float).reshape(3, 10, 1)
    scores = compute_isc(data, ["a", "b", "b"], channel="x")
    assert np.isnan(scores[0])
    assert np.isfinite(scores[1:]).all()


def test_legacy_scale_identity():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(8, 120, 1))
    groups = ["g"] * 8
    raw = compute_isc(data, groups, channel="x", transform="raw")
    legacy = compute_isc(data, groups, channel="x", transform="legacy")
    assert np.allclose(legacy, (1 + raw) / 2)


def test_simulation_seed_is_deterministic():
    config = {
        "seed": 20260816,
        "replicates": 2,
        "series_length": 120,
        "boundary_index": 60,
        "window_samples": 20,
        "ar1": [0.2],
        "group_size": [6],
        "shared_signal": [0.2],
        "affected_channel": ["x"],
        "alpha": 0.05,
    }
    first = simulate_recovery(config)
    second = simulate_recovery(config)
    assert first.equals(second)
