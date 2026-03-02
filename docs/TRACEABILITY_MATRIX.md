# Traceability Matrix

This matrix maps major claims to explicit code anchors and output artifacts.

Related docs:
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [EXPERIMENTS.md](./EXPERIMENTS.md)
- [RESULTS_ATLAS.md](./RESULTS_ATLAS.md)

## 1. Editorial Policy

1. Prefer strongest wording that is fully supported by code and artifacts.
2. Resolve claim conflicts by prioritizing raw artifact values over narrative boilerplate.
3. Keep claim language tied to observed setup (opponent pool, hyperparameters, run count).

Status legend:
- `Aligned`: code and artifacts support claim directly.
- `Qualified`: claim is true with scope limitations stated.
- `Reconcile`: wording needs adjustment to match current implementation/output.

## 2. Architecture Claims

| Claim ID | Claim | Code anchor(s) | Artifact anchor(s) | Status | Notes |
|---|---|---|---|---|---|
| A-01 | Standard IPD payoff matrix is R=3, S=0, T=5, P=1 | `ipd_core.py:26-31` | report section 1 | Aligned | Hardcoded matrix |
| A-02 | Strategy representation supports named callable and optional bitstring metadata | `ipd_core.py:34-43` | report section 2.1 | Aligned | Core abstraction |
| A-03 | Match simulation flips history perspective for opponent strategy | `ipd_core.py:76` | N/A (behavioral) | Aligned | Ensures symmetric decision context |
| A-04 | Tournament engine supports optional self-play | `ipd_core.py:90`, `ipd_core.py:100` | `tournament_results.csv` generation path | Aligned | Final tournament uses default include_self_play |
| A-05 | Bitstring encoding enforces `1 + 4^n` length | `ipd_core.py:220-223` | `memory_depth_results.csv` (`strategy_bits`) | Aligned | Complexity map consistent |
| A-06 | Shared evaluator caches match and fitness results globally with lock protection | `optimization.py:44-46`, `optimization.py:71-84`, `optimization.py:91-104`, `optimization.py:119-148` | runtime improvements visible indirectly in `time_taken` fields | Qualified | No direct cache-hit counter artifact |
| A-07 | Full-run orchestration writes stats, Pareto, ZD, convergence CI, metadata, and report | `main.py:187-225`, `main.py:228` | files in `results/` | Aligned | End-to-end pipeline confirmed |

## 3. Algorithm Claims

| Claim ID | Claim | Code anchor(s) | Artifact anchor(s) | Status | Notes |
|---|---|---|---|---|---|
| G-01 | GA uses roulette-wheel selection | `optimization.py:222-238` | `method_comparison.csv`, `ga_parameter_tuning.csv` | Aligned | Selection mechanics explicit |
| G-02 | GA uses single-point crossover | `optimization.py:240-254` | same as above | Aligned | Crossover point sampled in bit range |
| G-03 | GA uses bit-flip mutation | `optimization.py:256-264` | `ga_parameter_tuning.csv` | Aligned | Mutation-rate sensitivity measured |
| E-01 | EDA samples from probability vector and updates with learning-rate smoothing | `optimization.py:395-416` | `method_comparison.csv`, `memory_depth_results.csv` | Aligned | Distribution-based search |
| H-01 | HillClimbing performs local bit-flip neighbor search with restarts | `optimization.py:525-590` | `method_comparison.csv` | Aligned | Restart loop explicit |
| T-01 | TabuSearch uses tabu list and aspiration criterion | `optimization.py:695`, `optimization.py:709-720`, `optimization.py:730-732` | `method_comparison.csv` | Aligned | Aspiration accepts tabu if best-ever improved |
| O-01 | Evaluator supports tournament fitness and variance-penalized objective | `optimization.py:128-140` | protocol values in report + CSV outcomes | Aligned | Used widely in experiments |
| O-02 | Coevolution mode samples peer opponents | `optimization.py:165-169` | indirectly reflected in GA/EDA experiment outcomes | Qualified | Only enabled in some experiment families |

## 4. Experiment and Results Claims

| Claim ID | Claim | Code anchor(s) | Artifact anchor(s) | Status | Notes |
|---|---|---|---|---|---|
| X-01 | GA parameter tuning grid uses pop sizes 50/100/200 and mutation 0.001/0.01/0.05 | `experiments.py:116-117` | `ga_parameter_tuning.csv` | Aligned | 9 combinations present |
| X-02 | Memory-depth experiment evaluates depths 1-5 across all four methods | `experiments.py:163`, `experiments.py:168-226` | `memory_depth_results.csv` | Aligned | 20 rows |
| X-03 | Method comparison runs 5 repetitions with seeded variation | `experiments.py:237`, `experiments.py:257-263` | `method_comparison.csv` | Aligned | 20 rows |
| X-04 | Tournament uses evolved champion + reference strategies with 200 rounds | `experiments.py:370`, `experiments.py:438`, `experiments.py:442` | `tournament_results.csv` | Aligned | matches=10 per strategy |
| X-05 | ML experiment in full suite uses train sizes 500/1000/2000 at memory depth 2 | `experiments.py:773`, `experiments.py:459` | `ml_results.csv` | Aligned | 15 rows |
| X-06 | Pareto frontier extraction minimizes bits while maximizing fitness | `analysis.py:95-126` | `pareto_frontier.csv` | Aligned | Non-dominated points only |
| X-07 | Statistical analysis performs one-way ANOVA and pairwise effect sizes | `analysis.py:54-92` | `statistical_analysis.json` | Aligned | ANOVA + pairwise t-tests + Cohen's d |
| X-08 | ZD classification uses linear fit and `chi` thresholding | `zd_analysis.py:34-52`, `zd_analysis.py:72-79` | `zd_analysis.json` | Aligned | Approximate diagnostics |

## 5. Key Findings in `comprehensive_report.txt`

| Claim ID | Report finding | Artifact anchor(s) | Status | Documentation wording policy |
|---|---|---|---|---|
| K-01 | "Best Performing Method: EDA" | report section 3.3; `method_comparison.csv` means | Aligned | Keep, but include ANOVA p-value context |
| K-02 | "All methods evolved competitive strategies" | `tournament_results.csv`, `method_comparison.csv` | Qualified | Frame as "within tested pool and settings" |
| K-03 | "Memory depth 1 is sufficient for most scenarios" | `memory_depth_results.csv` | Reconcile | Data also shows higher peak at depth 3; rephrase as efficiency trade-off statement |
| K-04 | "RandomForest and GradientBoosting showed best performance" | report section 4.4 vs section 3.5 and `ml_results.csv` | Reconcile | Use artifact-backed result: NeuralNetwork highest mean F1 in this run |
| K-05 | "Population-based methods more consistent than local search" | `method_comparison.csv`, report section 3.3 | Qualified | Supported in current run; keep scope-limited |

## 6. Discrepancy Reconciliation Table

| Topic | Intended or narrative statement | Implemented behavior | Observed artifact state | Resolution guidance |
|---|---|---|---|---|
| Convergence plot filename | Report generated-files list includes `convergence_comparison.png` | Main post-processing and visualization use `convergence_ci.png` (`main.py:220`, `experiments.py:619`) | `results/convergence_ci.png` exists; `convergence_comparison.png` absent | Cite `convergence_ci.png` in docs/paper |
| ML best model narrative | Report section 4.4 highlights RF/GB | ML table generation computes grouped metrics from `ml_results.csv` | Report section 3.5 and CSV show NeuralNetwork best mean F1 | Use section 3.5 table and CSV values as canonical |
| "Depth 1 sufficient" phrasing | Narrative suggests depth>2 generally unnecessary | Experiment includes depths up to 5 and reports higher GA peak at depth 3 | `memory_depth_results.csv` peak is depth 3 GA | Reframe as fitness-complexity/runtime trade-off |
| Feature count wording in external blueprint history | Prior wording referenced 9 features | Current extractor emits 10 features including `tft_like` (`ml_prediction.py:76`) | `ml_results.csv` reflects trained models on this feature set | Use 10-feature description in docs |

## 7. Claim Coverage Summary

Coverage achieved:
- Architecture claims: 7 mapped.
- Algorithm claims: 8 mapped.
- Experiment/results claims: 8 mapped.
- Report key findings: 5 mapped.
- Reconciliation entries: 4 mapped.

Total mapped claim entries: 32.

## 8. Maintenance Rules for Future Edits

1. When changing code behavior, update this matrix in the same commit.
2. If artifact schemas change, update [INTERFACES_AND_DATA_CONTRACTS.md](./INTERFACES_AND_DATA_CONTRACTS.md) first, then adjust claim anchors.
3. Do not introduce narrative claims without direct code and artifact anchors.
