# Table schema

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| Table 1 | Dataset and provenance audit | dataset/experiment | available N, accepted N, sessions, channels, boundaries, missingness | manifests and validation reports | analysis pipeline |
| Table 2 | Primary crossed-null results | estimand | effect, interval, label p, time p, IUT p, evidence state | immutable primary results JSON | analysis pipeline |
| Table 3 | Benchmark and transport performance | fold/dataset | AUC, balanced accuracy, Brier, calibration, transported contrast | CV predictions and transfer results | analysis pipeline |
| Table S1 | Full specification grid | specification | effect, interval/p, direction | specification CSV | analysis pipeline |
| Table S2 | Simulation operating characteristics | truth state/scenario | bias, Type I error, coverage, power, classification rate | simulation CSV | simulation pipeline |
| Table S3 | Memory and EEG validation | model/outcome | coefficient, interval, p, N | secondary results CSV | analysis pipeline |
