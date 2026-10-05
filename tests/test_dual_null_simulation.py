import numpy as np

from boundaryisc.dual_null_simulation import one_dual_null_simulation


def test_dual_null_simulation_is_seed_reproducible():
    kwargs = dict(
        target_n=8,
        control_n=12,
        timepoints=500,
        boundaries=np.array([120, 240, 360]),
        width=20,
        rho=0.6,
        global_signal=0.5,
        common_boundary_signal=0.4,
        interaction_signal=0.8,
        smooth_nonstationarity=0.0,
        label_permutations=99,
        temporal_shifts=99,
    )
    first = one_dual_null_simulation(np.random.default_rng(3), **kwargs)
    second = one_dual_null_simulation(np.random.default_rng(3), **kwargs)
    assert first == second


def test_large_global_only_signal_does_not_create_boundary_effect():
    result = one_dual_null_simulation(
        np.random.default_rng(17),
        target_n=20,
        control_n=30,
        timepoints=1000,
        boundaries=np.array([200, 400, 600, 800]),
        width=40,
        rho=0.4,
        global_signal=2.0,
        common_boundary_signal=0.0,
        interaction_signal=0.0,
        smooth_nonstationarity=0.0,
        label_permutations=199,
        temporal_shifts=199,
    )
    assert result.p_global_label <= 0.01
    assert not result.p_boundary_iut <= 0.05


def test_common_boundary_response_is_not_group_interaction():
    result = one_dual_null_simulation(
        np.random.default_rng(27),
        target_n=30,
        control_n=74,
        timepoints=800,
        boundaries=np.array([120, 260, 400, 540, 680]),
        width=40,
        rho=0.5,
        global_signal=0.0,
        common_boundary_signal=2.0,
        interaction_signal=0.0,
        smooth_nonstationarity=0.0,
        label_permutations=499,
        temporal_shifts=499,
    )
    assert abs(result.boundary_effect) < 0.5
    assert result.p_boundary_iut > 0.05
