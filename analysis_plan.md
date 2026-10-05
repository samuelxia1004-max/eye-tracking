# Frozen analysis plan

Status: frozen 2026-08-20T03:38:12Z before inferential BBBD reanalysis and before any BBBD-to-Sherlock transported outcome was inspected. This is a local time-stamped analysis plan, not a public preregistration.

## Question and estimands

The study asks whether a previously reported Short-R eye-synchrony pattern is (a) a global cohort-relative difference, (b) unusually concentrated at the annotated event-boundary configuration, both, or neither. The primary boundary estimand is

`mean(boundary - pre)_Short-R - mean_group(mean(boundary - pre)_Long, Schema, Short-P)`.

The three control groups receive equal weight. Two evidential conditions are necessary for a boundary-specific group claim: the observed value must be unusual under fixed-size participant-label exchangeability and under common circular relocation of the complete boundary configuration. The intersection–union p-value is the maximum of the two component p-values. Both components and their null distributions are always reported.

## Data

### BBBD external benchmark

Use only official Zenodo record 10.5281/zenodo.19241964, Experiments 2 and 3. The archive documents 31 and 29 participants. Complete paired ocular data provide 31 and 28 participants respectively, 294 participant-video pairs and 588 attentive/distracted records. Session 1 is attentive viewing; session 2 is distracted re-viewing of the same videos in the same order. The target label is therefore a distracted-session benchmark confounded with repetition and order.

Read the final two visual-degree columns from each released gaze derivative and the first column from each released pupil derivative. Convert the released gaze/pupil interpolation intervals to the proportion of source samples outside interpolation within each 10-Hz temporal bin. Resample the released continuous derivative from 128 to 10 Hz using polyphase anti-alias filtering. A channel-window is eligible if its mean original-valid proportion is at least .50. Sensitivities are 0, .30 and .70. Correlations use the official interpolated signal in eligible windows.

Use non-overlapping 10-s windows. For each participant-video-session record and channel, correlate the record with the mean of other participants watching the same video. Pool references across the attentive and distracted session labels; labels are never used to construct a reference. Fisher transform correlations and average across eligible windows. Primary features are the mean X, Y and pupil Fisher-z ISC only.

### Sherlock transport archive

Use the previously frozen 10-Hz cache with 104 unique participants: Short-R 30, Long 25, Schema 26 and Short-P 23. Use X, Y and pupil, 12,900 samples, and the first 18 annotated boundaries that admit the complete analysis window. Corresponding raw validity, interpolation and calibration data are unavailable. The two 2025 source papers and this study reuse this commercial-film archive and are not independent replications.

## BBBD benchmark validation

Fit L2 logistic regression after standardizing features within each training fold. Outer evaluation crosses five participant-disjoint folds with held videos: a test record is predicted only by a model fitted with neither its participant nor its video. For the label-free reference, no held outer participant contributes; non-held, unlabeled records for the held video may form the stimulus-locked reference. This is transductive reference construction, not supervised training on the held video.

Select `C` from `[.001, .01, .1, 1, 10, 100]` by inner participant/video-blocked cross-validation; ties choose the smaller C. Report pooled out-of-fold ROC AUC, balanced accuracy, Brier score, calibration intercept/slope, and every fold result. The validation gate passes only if (1) the participant-mean attentive-minus-distracted out-of-fold probability difference has a two-sided 10,000-draw sign-flip p-value no greater than .05 and (2) a two-way participant/video bootstrap gives an AUC 95% interval whose lower endpoint exceeds .50. The two-way bootstrap is stratified by experiment and uses 5,000 draws. If the gate fails, do not fit or transport a final score.

If the gate passes, choose the modal outer-fold C, breaking ties toward stronger regularization, refit on all eligible BBBD pairs, serialize coefficients and scaling parameters, and make no change after Sherlock outcomes are available.

## EEG convergence

EEG is a secondary within-cohort convergence analysis, not part of the validation gate. Resample the released 64-channel processed BDF records to 16 Hz. For each video/session, fit three CorrCA components without the held participant fold and correlate each held participant’s component time courses with the training-participant mean. Join scores to ocular out-of-fold predictions by participant, video and condition. Estimate a condition-, participant- and video-adjusted association with two-way clustered uncertainty, plus the correlation between paired attentive-minus-distracted ocular and EEG changes. Do not describe this reused-cohort analysis as independent neural validation.

## Frozen Sherlock application

Construct a pooled leave-one-participant-out reference across all 104 participants without group labels. Apply the frozen BBBD scaling and coefficients to rolling 10-s X, Y and pupil Fisher-z correlations. The resulting logistic output is a transported cohort-relative score; it is not a calibrated probability of attention in Sherlock.

Primary global inference is the Short-R mean full-film score minus the equal-weight mean of the three other group means, with 10,000 fixed-size participant-label permutations and 5,000 group-weighted participant bootstrap draws.

For boundary inference, compare 10-s windows centered on each of 18 boundaries with the preceding 10-s windows. Run 10,000 fixed-size participant-label permutations. Independently run 10,000 nonzero common circular shifts of the complete boundary vector, preserving autocorrelation and boundary spacing. Use two-sided plus-one p-values. Uncertainty uses 5,000 participant-within-group and boundary Bayesian bootstrap draws. Report the four evidence states regardless of direction.

Secondary analyses are Short-R versus each group with Holm adjustment and group-adjusted associations of participant boundary change with `remember` and `detail`. They are associations, not causal effects or out-of-sample predictions.

## Specification audit

Report every planned raw-ISC cell crossing channel (X, Y, pupil, gaze-only, equal-weight composite), estimator (pairwise, leave-one-out), aggregation (raw r, Fisher z), window (5, 10, 20 s) and reference (within group, all participants), plus the learned transported-score result at 5, 10 and 20 s. Do not use an arbitrary percentage of concordant specifications as a pass/fail criterion.

## Simulations

Generate AR(1) participant panels under four states: neither; a global target-group offset only; a boundary response common to both groups only; and a target-group-by-boundary interaction. The crossed-null test should reject only in the fourth state, while the separate global test should detect the second. One-factor scenarios vary autocorrelation, smooth group-specific non-stationarity, participant count, boundary count and signal strength. Use seed 20260816, 250 replicates per state/scenario, and 199 label and temporal randomizations per replicate. Report detection rates with Wilson intervals and both component-test rates.

**D10 amendment (2026-08-20):** The local freeze mistakenly encoded 20260820, although the author-approved implementation plan specified 20260816. After the discrepancy was found during manuscript integration, both configuration files were restored to 20260816 and every stochastic empirical analysis, simulation, table, source-data file, figure and invariant was regenerated. No alternative seed result was selected for reporting.

## Error control, language and stopping rules

All empirical randomization tests are two-sided and use `(b + 1)/(m + 1)`. Alpha is .05. The gate, estimands, directions, windows, feature set and nulls may not be changed in response to results. Implementation errors discovered after freeze are corrected only with a dated deviation describing impact and rerunning all affected outputs. No conclusion equates eye synchrony, pupil response or the transported score with attention, event segmentation, pleasure or neural damage. Null and discordant results remain in the paper.
