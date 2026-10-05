# Figure data manifest

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| Figure 1 | `results/source_data/figure1_framework.csv` | real definitions + simulation examples | frozen protocol | `scripts/figures/figure1_framework.py` | SVG, PDF, PNG |
| Figure 2 | `results/source_data/figure2_bbbd_benchmark.csv` | real | BBBD cross-validation and EEG validation | `scripts/figures/figure2_bbbd_benchmark.py` | SVG, PDF, PNG |
| Figure 3 | `results/source_data/figure3_sherlock_transfer.csv` | real | Sherlock global/boundary analyses | `scripts/figures/figure3_sherlock_transfer.py` | SVG, PDF, PNG |
| Figure 4 | `results/source_data/figure4_simulations.csv` | real synthetic-study outputs | simulation pipeline | `scripts/figures/figure4_simulations.py` | SVG, PDF, PNG |

No mock numerical result may appear in a submission artifact. Every figure source table must carry input hashes and generation metadata.
