# Data dictionary

## Frozen inputs

| Object | Unit / shape | Fields | Interpretation boundary |
|---|---|---|---|
| BBBD ocular archive records | participant × video × session × time | released gaze X, gaze Y, pupil | Session is attentive first or distracted re-viewing second; signals are released derivatives, not raw calibration records. |
| BBBD EEG archive records | participant × video × session × time × 64 channels | released preprocessed BDF | Used only for within-cohort convergent association. |
| Sherlock ocular cache | 104 participants × 12,900 10-Hz samples × 3 channels | X, Y, pupil | Processed public archive; no raw quality masks or video redistribution. |
| Sherlock metadata | participant | group, remember, detail | Four randomized exposure groups; memory outcomes are secondary associations. |
| Sherlock boundary vector | 18 primary event locations | sample index at 10 Hz | Nineteen published locations exist; the last cannot accommodate the complete post-window and is excluded by the inherited rule. |

## Primary BBBD feature table

| Column | Type | Definition |
|---|---|---|
| `record_index` | integer | Stable cache row identifier. |
| `participant` | string | Experiment-qualified participant identifier. |
| `video` | string | Experiment-qualified stimulus identifier. |
| `condition` | categorical | `attentive` or `distracted`; condition labels do not enter reference construction. |
| `n_windows` | integer | Count of non-overlapping windows available on the padded video timeline. |
| `isc_x_mean` | float | Mean Fisher-z correlation between X and a label-free, other-participant video reference. |
| `isc_y_mean` | float | Mean Fisher-z correlation between Y and the corresponding reference. |
| `isc_pupil_mean` | float | Mean Fisher-z correlation between pupil and the corresponding reference. |
| `isc_*_sd` | float | Across-window dispersion; audited but excluded from the primary model. |

## Outer-prediction table

| Column | Definition |
|---|---|
| `participant_fold` | Participant-disjoint outer fold. |
| `held_video` | Entire stimulus excluded from supervised fitting. |
| `probability_attentive` | Out-of-fold logistic probability for the attentive-session label. It is a benchmark score, not a direct attention probability in a new domain. |
| `truth_attentive` | Binary BBBD session label. |

## Sherlock transported-score tables

| Column | Definition |
|---|---|
| `global_score` | Frozen BBBD model applied to full-film cohort-relative Sherlock ISC features. |
| `boundary_change` | Participant mean of centered-boundary minus pre-boundary transported score. |
| `effect` / `observed` | Short-R mean minus the equal-weight mean of the three control-group means. |
| `label_p_value` | Fixed-size participant-label randomization p-value. |
| `temporal_p_value` | Common circular shift of the complete boundary configuration. |
| `intersection_union_p` | Maximum of the two component p-values. |

## Simulation table

`scenario` identifies a one-factor deviation from baseline. `state` is one of `neither`, `global_group_only`, `common_boundary_only`, or `group_boundary_interaction`. Detection columns are decisions at the frozen alpha; component p-values remain available so the conjunctive result cannot conceal failure of either condition.
