# Figure contracts

Backend for every figure: Python (Matplotlib). Target: Behavior Research Methods, double-column 183 mm unless noted. Exports: editable SVG, PDF, and 600-dpi PNG. All quantitative panels require tidy CSV source data.

## Figure 1 — Design and inferential claims

Core conclusion: A boundary-specific group claim requires two distinct randomization targets, and the external ocular benchmark is frozen before Sherlock outcomes are read.

Figure archetype: schematic-led composite.

Final size: 183 × 150 mm.

Panel map:

- a: BBBD attentive/distracted paired-video benchmark and crossed participant/video holdout.
- b: outcome-blind transport to Sherlock.
- c: participant-label null for the group estimand.
- d: common-shift boundary-location null for the same estimand.
- e: four evidence states and intersection–union decision.

Evidence hierarchy: method definition is primary; dataset counts and non-overlap rules are validation; limitations are embedded annotations.

Statistics needed: sample units and fold definitions only; no p-values.

Source data needed: frozen manifests and configuration.

Image-integrity notes: diagrammatic only; no generated scientific imagery.

Reviewer risk: the two tests could appear redundant, or the external score could appear to use Sherlock labels. Make the null targets and freeze boundary explicit.

## Figure 2 — External ocular benchmark

Core conclusion: The BBBD ocular score either discriminates the paired session conditions under participant-and-video holdout and converges with EEG, or it fails the frozen transport gate.

Figure archetype: quantitative grid with a dominant held-out performance panel.

Final size: 183 × 150 mm.

Panel map:

- a: outer-fold ROC with two-level uncertainty and chance line.
- b: paired attentive–distracted probability differences by participant/video.
- c: calibration curve with Brier score.
- d: channel coefficients or ablation estimates.
- e: held-out ocular-score association with EEG ISC, if available; otherwise a declared unavailable/failed validation panel is moved to supplement.

Evidence hierarchy: held-out AUC and paired differences are primary; calibration and ablation are validation; EEG is convergent evidence.

Statistics needed: train/test blocking, n participants, n videos, interval method, label-permutation p-value, gate decision.

Source data needed: outer predictions, fold metrics, bootstrap draws, coefficient table, EEG validation table.

Image-integrity notes: none.

Reviewer risk: repetition/order confounding and transductive reference construction. State both in the panel notes.

## Figure 3 — Sherlock global and boundary evidence

Core conclusion: A global transported ocular-score difference and annotated-boundary specificity are separable empirical results.

Figure archetype: asymmetric mixed quantitative figure.

Final size: 183 × 160 mm.

Panel map:

- a: global transported-score distribution and Short-R contrast.
- b: boundary-minus-pre group contrast with participant uncertainty.
- c: participant-label null distribution.
- d: common-shift temporal null distribution.
- e: evidence-state summary with the larger component p-value.
- f: channel/estimator/window specification curve.

Evidence hierarchy: frozen global and boundary estimands are primary; component nulls are co-primary for boundary specificity; specification audit is supporting.

Statistics needed: N by group, exact estimand, two-sided plus-one p-values, bootstrap interval, no threshold-based robustness claim.

Source data needed: participant scores, group summaries, label null, temporal null, specification grid.

Image-integrity notes: none.

Reviewer risk: eye synchrony may be overread as attention or event segmentation. Use “transported ocular score” and “boundary specificity,” not mechanistic labels.

## Figure 4 — Recovery simulation

Core conclusion: The crossed decision suppresses boundary-specific claims under neither, global-only and common-boundary-only states, while detecting group-by-boundary interactions with scenario-dependent power.

Figure archetype: quantitative grid.

Final size: 183 × 135 mm.

Panel map:

- a: four truth states.
- b: detection rates for global, boundary-label, temporal, and intersection–union decisions.
- c: evidence-state confusion matrix.
- d: sensitivity to autocorrelation, signal strength, sample size, boundary count and smooth non-stationarity.

Evidence hierarchy: Type I error under each composite null is primary; power and degradation are validation.

Statistics needed: simulation replicates, seed, parameter grid, Monte Carlo interval.

Source data needed: replicate-level simulation CSV and summary CSV.

Image-integrity notes: simulated data must be visibly labeled as simulation.

Reviewer risk: a stylized simulation can be mistaken for calibration to the observed effect. Parameters and non-calibration must be stated.
