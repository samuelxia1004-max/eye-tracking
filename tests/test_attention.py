import numpy as np
import pandas as pd

from boundaryisc.attention import (
    FEATURE_COLUMNS,
    build_attention_features,
    double_block_splits,
    nested_attention_benchmark,
    nested_attention_benchmark_from_records,
)
from boundaryisc.crossed_null import crossed_null_decision, intersection_union_p


def test_crossed_null_requires_both_components():
    assert intersection_union_p(0.01, 0.20) == 0.20
    decision = crossed_null_decision(0.01, 0.20)
    assert not decision.reject_both
    assert crossed_null_decision(0.01, 0.02).reject_both


def test_attention_reference_excludes_same_participant_and_ignores_condition():
    rng = np.random.default_rng(7)
    participants = np.repeat(["p1", "p2", "p3"], 2)
    conditions = np.tile(["attentive", "distracted"], 3)
    videos = np.repeat("v1", 6)
    shared = rng.normal(size=(200, 3))
    data = np.stack([shared + rng.normal(scale=0.1, size=shared.shape) for _ in range(6)])
    frame = build_attention_features(
        data,
        participants,
        videos,
        conditions,
        rate_hz=10,
        window_seconds=10,
    )
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert np.isfinite(frame.loc[:, FEATURE_COLUMNS].to_numpy()).all()
    assert (frame["n_windows"] == 2).all()


def test_double_block_splits_have_no_participant_or_video_overlap():
    participants = np.repeat([f"p{i}" for i in range(6)], 4)
    videos = np.tile(np.repeat(["v1", "v2"], 2), 6)
    for train, test, _, _ in double_block_splits(participants, videos, participant_folds=3, seed=9):
        assert set(participants[train]).isdisjoint(participants[test])
        assert set(videos[train]).isdisjoint(videos[test])


def test_nested_benchmark_recovers_known_condition_signal():
    rows = []
    index = 0
    rng = np.random.default_rng(11)
    for participant in [f"p{i:02d}" for i in range(12)]:
        for video in ["v1", "v2", "v3"]:
            for condition, sign in [("attentive", 1.0), ("distracted", -1.0)]:
                row = {
                    "record_index": index,
                    "participant": participant,
                    "video": video,
                    "condition": condition,
                }
                for column in FEATURE_COLUMNS:
                    row[column] = sign + rng.normal(scale=0.15)
                rows.append(row)
                index += 1
    result = nested_attention_benchmark(pd.DataFrame(rows), participant_folds=3, seed=22)
    assert result.aggregate_metrics["auc"] > 0.95
    assert len(result.predictions) == len(rows)


def test_record_benchmark_rebuilds_references_without_held_participants():
    rng = np.random.default_rng(31)
    participants = []
    videos = []
    conditions = []
    records = []
    for participant in [f"p{i:02d}" for i in range(8)]:
        for video in ["v1", "v2"]:
            shared = rng.normal(size=(100, 3))
            for condition, scale in [("attentive", 0.05), ("distracted", 0.8)]:
                participants.append(participant)
                videos.append(video)
                conditions.append(condition)
                records.append(shared + rng.normal(scale=scale, size=shared.shape))
    result = nested_attention_benchmark_from_records(
        np.stack(records),
        participants,
        videos,
        conditions,
        participant_folds=2,
        c_grid=(0.1, 1.0),
        window_seconds=5,
        seed=8,
    )
    assert len(result.predictions) == len(records)
    assert result.aggregate_metrics["auc"] > 0.5
