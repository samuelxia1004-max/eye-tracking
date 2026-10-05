import numpy as np
import pandas as pd

from boundaryisc.attention_inference import (
    FrozenOcularModel,
    boundary_location_test,
    boundary_probability_contrast,
    participant_signflip_test,
    group_weighted_bootstrap,
    participant_boundary_bootstrap,
    rolling_loo_features,
)


def test_participant_signflip_detects_paired_probability_direction():
    rows = []
    for participant in [f"exp2_sub-{index:02d}" for index in range(12)]:
        for video in ["exp2_stim-01", "exp2_stim-02"]:
            rows.append(
                {
                    "participant": participant,
                    "video": video,
                    "condition": "attentive",
                    "probability_attentive": 0.8,
                }
            )
            rows.append(
                {
                    "participant": participant,
                    "video": video,
                    "condition": "distracted",
                    "probability_attentive": 0.2,
                }
            )
    result = participant_signflip_test(pd.DataFrame(rows), n_permutations=999, seed=4)
    assert result["p_value"] <= 0.01


def test_rolling_loo_features_recover_shared_signal():
    rng = np.random.default_rng(12)
    shared = rng.normal(size=(400, 3))
    data = np.stack([shared + rng.normal(scale=0.05, size=shared.shape) for _ in range(8)])
    features = rolling_loo_features(data, width=50)
    assert features.shape == data.shape
    assert np.nanmean(features) > 1.0


def test_boundary_contrast_and_location_test_known_effect():
    groups = np.asarray(["target"] * 10 + ["control"] * 14)
    curve = np.zeros((24, 500), dtype=float)
    boundaries = [100, 200, 300, 400]
    width = 20
    for boundary in boundaries:
        curve[:10, boundary - width // 2] = 1.0
    effect, _, _ = boundary_probability_contrast(
        curve, groups, boundaries, width=width, target="target"
    )
    assert effect > 0.5
    result = boundary_location_test(
        curve,
        groups,
        boundaries,
        width=width,
        target="target",
        n_shifts=499,
        seed=2,
    )
    assert result["p_value"] <= 0.05
    global_bootstrap = group_weighted_bootstrap(
        np.r_[np.ones(10), np.zeros(14)],
        groups,
        target="target",
        n_bootstrap=199,
        seed=3,
    )
    assert global_bootstrap["ci_low"] > 0
    boundary_bootstrap = participant_boundary_bootstrap(
        curve,
        groups,
        boundaries,
        width=width,
        target="target",
        n_bootstrap=199,
        seed=3,
    )
    assert boundary_bootstrap["ci_low"] > 0
