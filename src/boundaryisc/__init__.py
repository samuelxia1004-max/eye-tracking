"""Public interface for the boundaryisc package."""

from .core import boundary_contrast, compute_isc
from .attention import (
    build_attention_features,
    double_block_splits,
    nested_attention_benchmark,
    nested_attention_benchmark_from_records,
)
from .crossed_null import intersection_union_p, crossed_null_decision
from .inference import pseudo_boundary_test
from .simulation import simulate_recovery
from .specification import run_specification_grid

__all__ = [
    "compute_isc",
    "boundary_contrast",
    "build_attention_features",
    "double_block_splits",
    "nested_attention_benchmark",
    "nested_attention_benchmark_from_records",
    "intersection_union_p",
    "crossed_null_decision",
    "pseudo_boundary_test",
    "run_specification_grid",
    "simulate_recovery",
]

__version__ = "0.2.0"
