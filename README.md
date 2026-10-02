# How Much Network Does Early Lane-Change Recognition Need?

**A bi-objective neural architecture search on NGSIM data** — Joshua Mohebban, Politecnico di Milano

A small, self-contained study that applies a complete bi-objective neural architecture search (NAS) to a real, imbalanced prediction task, and examines how reliable its conclusions are.

**Task.** Given the last 3 s of a vehicle's motion (NGSIM I-80 trajectories), predict whether the vehicle will cross into another lane within the next 3 s. About 2% of the windows are positive.

**Search.** Over a space of 780 multilayer perceptrons (1–4 hidden layers, widths in {16, 32, 64, 128, 256}), random search maximises the validation AUC and minimises the number of parameters. The result is compared with a hand-designed baseline, `[64, 64]`.

The full description is in the report: [`report.pdf`](https://yoshuuqq.github.io/lane-change-nas/report.pdf).

## Main results

| Architecture | Parameters | Test AUC | Test AP |
|---|---|---|---|
| `64-64` (hand-designed baseline) | 11,969 | 0.8918 ± 0.0008 | 0.328 ± 0.004 |
| `16-32-16-16` | 3,297 | 0.9025 ± 0.0025 | 0.431 ± 0.007 |
| `16-128-16-128` | 8,481 | 0.9022 ± 0.0016 | 0.457 ± 0.004 |
| `256-256-256-256` (largest in the space) | 228,609 | 0.8916 ± 0.0015 | 0.348 ± 0.012 |

Mean ± standard error over five seeds, on a held-out test set used once.

- **The baseline is dominated by smaller networks.** A network 3.6 times smaller is better on both AUC and average precision (AP); a network with 30% fewer parameters improves the AP by 39%.
- **Larger networks buy nothing.** Beyond about 8,500 parameters nothing is gained: a network with 19 times the parameters of the baseline is no better than it.
- **One training run is not enough to rank architectures.** The variability between runs with different seeds is of the same order as the differences between architectures, and selecting the best single-run result overestimates it. The candidates were therefore re-evaluated with five seeds before computing the Pareto front.

## Contents

| File | Content |
|---|---|
| `report.pdf` | The report |
| `01_data.ipynb` | Exploratory analysis, split by vehicle, window extraction, normalisation |
| `02_baseline_preliminary.ipynb` | First hand-written training of the baseline (preliminary; not used for the reported numbers) |
| `03_nas.ipynb` | All training runs: search (300 architectures), seed study, re-evaluation with five seeds, test |
| `04_test_analysis.ipynb` | Test tables, ROC and precision–recall curves, control on straddling windows |
| `05_verify_results.ipynb` | Recomputes every number of the Results section from the saved files, without training |
| `risultati_nas.csv`, `risultati_nas_v2.csv` | Search results (seed 0), before and after the early-stopping correction |
| `rumore_semi.csv` | Seed study: 6 architectures × seeds 0–4 |
| `multi_seed.csv` | Re-evaluation: 30 candidates × seeds 1–5 |
| `risultati_test.csv`, `punteggi_test.npz` | Test metrics and scores: 9 architectures × seeds 1–5 |

The notebooks are stored with their outputs, so the results can be read without running anything. Comments and explanations are in English; identifiers, column names and printed messages are in Italian, and each notebook opens with a short glossary.

## Reproducing the results

Requirements: Python 3.9 or later and the packages in `requirements.txt`.

```
pip install -r requirements.txt
jupyter notebook
```

**Checking the reported numbers** needs no data and no training: run `05_verify_results.ipynb`, which reads the result files included here. `04_test_analysis.ipynb` works in the same way.

**Running everything from scratch:**

1. Download the NGSIM I-80 trajectory data (file `trajectories-0400-0415.csv`, period 4:00–4:15 p.m.) from the [U.S. Department of Transportation](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) and place it in this folder. The raw file is not included here because of its size.
2. Run `01_data.ipynb`, which creates `dataset_finestre.npz`.
3. Run `03_nas.ipynb`. The results are saved after every training run and existing result files are reused, so delete the `.csv` and `.npz` result files first to repeat the experiments. On an Apple M1 Pro the search takes about 70 minutes and the re-evaluation about 25.
4. Run `04_test_analysis.ipynb` and `05_verify_results.ipynb`.

Note on the data: the study uses the first 1,048,575 rows of the file (1,725 vehicles), because the copy used was truncated at that length; all reported results refer to this subset. Reading the full file would give a larger dataset and different numbers.

Training was run on the GPU of an Apple M1 Pro (PyTorch, MPS backend). On that machine, runs with the same seed gave identical results.

## Use of AI tools

An AI assistant (Claude, Anthropic) was used throughout this project. It drafted most of the code and proposed drafts of the text. All decisions on the study were taken by the author, who also ran all experiments, verified the results against the code and its outputs, and reviewed and revised the code and the text.
Moreover the code present in this repository is a cleaned-up version of the original one. Using Claude, comments and explanations were rewritten in english and the cell were reordered for readability. The executable code is unchanged, and the stored outputs are those of the original code.
