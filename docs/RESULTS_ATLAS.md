# Results Atlas

This atlas explains every artifact in `results/`, how to read it, and which claims each artifact can support.

Primary evidence folder:
- `results/`

Companion report:
- `results/comprehensive_report.txt`

Related docs:
- [EXPERIMENTS.md](./EXPERIMENTS.md)
- [TRACEABILITY_MATRIX.md](./TRACEABILITY_MATRIX.md)
- [PAPER_COMPANION.md](./PAPER_COMPANION.md)

## 1. Artifact Inventory (Current Run)

## 1.1 Data and text artifacts
- `comprehensive_report.txt`
- `ga_parameter_tuning.csv`
- `memory_depth_results.csv`
- `method_comparison.csv`
- `ml_results.csv`
- `pareto_frontier.csv`
- `reproducibility_metadata.json`
- `statistical_analysis.json`
- `tournament_results.csv`
- `zd_analysis.json`

## 1.2 Plot artifacts
- `convergence_ci.png`
- `evolved_comparison.png`
- `ga_parameter_heatmap.png`
- `memory_depth_impact.png`
- `method_comparison.png`
- `ml_comparison.png`
- `pareto_frontier.png`
- `tournament_results.png`

## 2. File-by-File Interpretation

| Artifact | Generated from | What it directly encodes | Safe interpretation | Not supported by this file alone |
|---|---|---|---|---|
| `ga_parameter_tuning.csv` | `experiments.run_parameter_tuning_ga` | GA grid outcomes for tested pop/mutation combos | Which tested GA configs performed best in this setup | Universal best GA settings |
| `ga_parameter_heatmap.png` | `create_visualizations` | Heatmap view of table above | Visual sensitivity to GA hyperparameters | Causal explanation for why one cell is best |
| `memory_depth_results.csv` | `experiments.run_memory_depth_experiment` | Method/depth fitness+runtime+bit complexity | Trade-offs of depth and optimizer under current settings | General depth theorem across all IPD scenarios |
| `memory_depth_impact.png` | `create_visualizations` | Line chart of depth vs best fitness by method | Relative directional trends by method | Statistical significance by itself |
| `pareto_frontier.csv` | `analysis.pareto_frontier_analysis` | Non-dominated rows over fitness vs bits | Efficient frontier under current 2-objective framing | Any objective beyond fitness+complexity |
| `pareto_frontier.png` | `plot_pareto_frontier` | Scatter + highlighted frontier points | Which sampled points are Pareto-optimal | That omitted points are always inferior in all contexts |
| `method_comparison.csv` | `experiments.compare_all_methods` | Per-run method outcomes | Run-to-run stability and spread | Final method ranking without stats |
| `method_comparison.png` | `create_visualizations` | Boxplot of best_fitness by method | Distribution contrast between methods | Why distributions differ |
| `convergence_ci.png` | `plot_convergence_ci` | Mean convergence with 95% CI bands | Relative convergence behavior over iterations | Absolute guarantee of convergence characteristics |
| `tournament_results.csv` | `experiments.evolved_vs_reference` | Ranking of champion evolved strategy + references | Relative performance in this tournament pool | Global invincibility |
| `tournament_results.png` | `create_visualizations` | Horizontal score bars by strategy | Rank visibility and score gaps | Transferability to unseen opponents |
| `evolved_comparison.png` | `create_visualizations` | Aggregate mean score by type (`Evolved` vs `Reference`) | Group-level comparison | Method-specific attribution inside evolved group |
| `ml_results.csv` | `experiments.run_ml_experiments` | Classifier metrics across train sizes | Predictive signal under current labels/features | General ML superiority in other feature spaces |
| `ml_comparison.png` | `create_visualizations` | Metric curves by model and train size | Relative metric trajectories | Robust calibration or uncertainty analysis |
| `statistical_analysis.json` | `analysis.statistical_significance_test` | ANOVA, pairwise tests, effect sizes, CIs | Quantified statistical context for method differences | Strong claims if run count is small |
| `zd_analysis.json` | `zd_analysis.analyze_zd_properties` | Fitted `chi`, `phi`, and classification labels | Approximate generous/extortionate/non-ZD categorization | Formal symbolic ZD proofs |
| `reproducibility_metadata.json` | `ExperimentRunner.log_reproducibility_metadata` | Environment and seed policy metadata | Provenance and reproducibility context | Full deterministic replay guarantee on every machine |
| `comprehensive_report.txt` | `main.generate_comprehensive_report` | Narrative summary and table snapshots | Human-readable synthesis baseline | Guaranteed consistency with all code paths (must verify against raw artifacts) |

## 3. Plot Reading Guide

## 3.1 `ga_parameter_heatmap.png`
Read across:
1. Rows = population size.
2. Columns = mutation rate.
3. Color = best fitness.

Use with:
- `ga_parameter_tuning.csv` for exact numeric values.

## 3.2 `memory_depth_impact.png`
Read across:
1. X-axis = memory depth.
2. Y-axis = best fitness.
3. Separate lines per optimization method.

Use with:
- `memory_depth_results.csv` for runtime and bit complexity context.

## 3.3 `method_comparison.png`
Read across:
1. One box per method.
2. Median and spread of `best_fitness` across 5 runs.

Use with:
- `method_comparison.csv` and `statistical_analysis.json` for significance context.

## 3.4 `convergence_ci.png`
Read across:
1. Line = method mean convergence trajectory.
2. Shaded area = 95% CI using `1.96 * SEM` (`experiments.py:662-691`).

## 3.5 `tournament_results.png`
Read across:
1. Bar length = average score in tournament.
2. Color encodes evolved vs reference.

Use with:
- `tournament_results.csv` for exact ranking values.

## 3.6 `evolved_comparison.png`
Read across:
1. Category-level averages by `type`.
2. Intended as a high-level summary, not detailed strategy attribution.

## 3.7 `ml_comparison.png`
Read across:
1. Four panels: accuracy, precision, recall, F1.
2. X-axis = train size, line per model.

Use with:
- `ml_results.csv` for exact numbers and per-size comparisons.

## 3.8 `pareto_frontier.png`
Read across:
1. Scatter points = all method/depth combinations.
2. Star markers = non-dominated frontier points.

Use with:
- `pareto_frontier.csv` for exact selected rows.

## 4. Numeric Highlights Table (Current Run)

| Metric | Value | Source |
|---|---:|---|
| Best GA tuning fitness | 257.71473741287303 | `ga_parameter_tuning.csv` |
| Best GA tuning config | pop=200, mut=0.05 | `ga_parameter_tuning.csv` |
| Peak memory-depth fitness | 266.12014479031325 (GA depth 3) | `memory_depth_results.csv` |
| Method comparison top mean | 251.559653 (EDA) | `method_comparison.csv`, `comprehensive_report.txt` |
| Tournament top rank | TF2T, avg 538.4 | `tournament_results.csv` |
| Evolved champion rank | 3rd (avg 534.8) | `tournament_results.csv` |
| Best mean ML model (report summary) | NeuralNetwork F1 = 0.344255 | `comprehensive_report.txt` section 3.5 |
| Best single ML row | NeuralNetwork F1 = 0.4461538461538462 @ train=2000 | `ml_results.csv` |
| ANOVA p-value | 0.5934853222459717 | `statistical_analysis.json` |
| ZD classification summary | 4/4 classified Generous | `zd_analysis.json` |

## 5. Cross-Artifact Consistency Map

| Claim type | Primary evidence | Corroborating evidence |
|---|---|---|
| Hyperparameter sensitivity | `ga_parameter_tuning.csv` | `ga_parameter_heatmap.png`, report section 3.1 |
| Depth-performance tradeoff | `memory_depth_results.csv` | `memory_depth_impact.png`, `pareto_frontier.csv`, report section 3.2 |
| Method comparative stability | `method_comparison.csv` | `method_comparison.png`, `convergence_ci.png`, `statistical_analysis.json` |
| Tournament competitiveness | `tournament_results.csv` | `tournament_results.png`, `evolved_comparison.png`, report section 3.4 |
| ML predictive signal | `ml_results.csv` | `ml_comparison.png`, report section 3.5 |
| Reproducibility metadata | `reproducibility_metadata.json` | `experiments.py:71-99`, `main.py:223-225` |

## 6. Interpretation Boundaries

Use these rules when writing conclusions from `results/`:
1. Treat observed rankings as configuration-dependent empirical outcomes.
2. Separate descriptive claims (what happened) from causal claims (why it happened).
3. Use statistical outputs before asserting method superiority.
4. Use "classified as" wording for ZD diagnostics, not formal proof language.
5. Prefer raw CSV/JSON values over narrative text when discrepancies exist.

## 7. Known Reconciliation Items

1. Report generated-files section references `convergence_comparison.png`, but pipeline currently writes `convergence_ci.png`.
2. Report section 4.4 says RandomForest and GradientBoosting performed best, while section 3.5 and `ml_results.csv` show NeuralNetwork with highest mean F1 in this run.

Documentation and paper writing should cite raw artifacts to resolve these conflicts.

## 8. Fast Citation Shortlist

For paper Methods:
- `experiments.py:703-789`
- `optimization.py:175-756`
- `ml_prediction.py:94-228`

For paper Results:
- `results/ga_parameter_tuning.csv`
- `results/memory_depth_results.csv`
- `results/method_comparison.csv`
- `results/tournament_results.csv`
- `results/ml_results.csv`
- `results/statistical_analysis.json`

For paper Reproducibility:
- `results/reproducibility_metadata.json`
- `main.py:169-226`
