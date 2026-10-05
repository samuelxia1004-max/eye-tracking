# `boundaryisc` public API and CLI

## Public Python functions

- `compute_isc(...)`: channel-wise intersubject correlation under explicit estimator, transform and reference rules.
- `boundary_contrast(...)`: pre/boundary/post contrast with participant-level outputs.
- `pseudo_boundary_test(...)`: autocorrelation-preserving temporal surrogate inference.
- `run_specification_grid(...)`: complete prespecified estimator/channel/window/reference grid.
- `simulate_recovery(...)`: legacy estimator-recovery simulation retained for backward compatibility.
- `build_attention_features(...)`: label-free, cohort-relative ocular feature construction.
- `nested_attention_benchmark_from_records(...)`: crossed participant/video validation with nested regularization selection.
- `crossed_null_decision(...)`: intersection–union decision from the label and temporal component tests.

## CLI

All inferential commands require a configuration whose status begins with `frozen` and whose input paths exist.

```text
boundaryisc --config config/analysis.json validate-config
boundaryisc --config config/analysis.json reproduce
boundaryisc --config config/analysis.json analyze
boundaryisc --config config/analysis.json simulate
boundaryisc --config config/analysis.json report
```

`reproduce` creates the verified 10-Hz BBBD cache. `analyze` executes the validation gate, optional EEG convergence analysis and (only after a passed gate) frozen Sherlock transport. `simulate` runs the four-state crossed-null operating-characteristic study. `report` regenerates deterministic tables and figures from result artifacts without refitting models.
