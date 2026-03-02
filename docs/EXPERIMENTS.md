# Experiments Reference

This document is the definitive reference for how experiments are configured, what each artifact means, and what conclusions are and are not supported by current outputs in `results/`.

Related docs:
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [INTERFACES_AND_DATA_CONTRACTS.md](./INTERFACES_AND_DATA_CONTRACTS.md)
- [RESULTS_ATLAS.md](./RESULTS_ATLAS.md)
- [REPRODUCIBILITY_AND_PROVENANCE.md](./REPRODUCIBILITY_AND_PROVENANCE.md)
- [TRACEABILITY_MATRIX.md](./TRACEABILITY_MATRIX.md)

## 1. Experiment Suite at a Glance

| Experiment family | Core function | Primary objective | Output artifacts |
|---|---|---|---|
| GA parameter tuning | `ExperimentRunner.run_parameter_tuning_ga` (`experiments.py:101`) | Measure GA sensitivity to population size and mutation rate | `ga_parameter_tuning.csv`, `ga_parameter_heatmap.png` |
| Memory-depth impact | `ExperimentRunner.run_memory_depth_experiment` (`experiments.py:150`) | Quantify fitness/runtime/complexity trade-offs across depths 1-5 | `memory_depth_results.csv`, `memory_depth_impact.png`, `pareto_frontier.csv`, `pareto_frontier.png` |
| Multi-method comparison | `ExperimentRunner.compare_all_methods` (`experiments.py:235`) | Compare GA, EDA, HillClimbing, TabuSearch across seeded runs | `method_comparison.csv`, `method_comparison.png`, `convergence_ci.png`, `statistical_analysis.json` |
| Evolved vs reference tournament | `ExperimentRunner.evolved_vs_reference` (`experiments.py:355`) | Benchmark best evolved champion against handcrafted baselines | `tournament_results.csv`, `tournament_results.png`, `evolved_comparison.png`, `zd_analysis.json` |
| ML scaling | `ExperimentRunner.run_ml_experiments` (`experiments.py:457`) | Measure classifier quality as training size increases | `ml_results.csv`, `ml_comparison.png` |

Full suite execution path:
- `run_full_experiment_suite()` in `experiments.py:703-789`.
- Called by `main.run_full_experiments()` in `main.py:169-226`.

## 2. Canonical Opponent Policy and Comparability

Canonical set:
- `STANDARD_OPPONENTS = [TFT, ALLD, ALLC, TF2T, STFT, GRIM, PAVLOV]` (`experiments.py:33`).

Policy:
- Helper `_get_standard_opponents` enforces this set and prints override info if a caller passes a different list (`experiments.py:50-60`).

Why this matters:
1. Fitness scores from different experiment families remain comparable because opponent composition is fixed.
2. Drift in opponent pools is prevented when experiments are invoked from different contexts.

Exception path:
- `evolved_vs_reference` optimizes against full `REFERENCE_STRATEGIES` for final champion selection (`experiments.py:369`, `experiments.py:438`).

## 3. Exact Protocols Per Experiment

### 3.1 GA parameter tuning
Source: `experiments.py:101-147`

Objective:
- Search GA hyperparameter grid and record best fitness and runtime.

Independent variables:
- `population_size`: `[50, 100, 200]` (`experiments.py:116`)
- `mutation_rate`: `[0.001, 0.01, 0.05]` (`experiments.py:117`)

Fixed settings:
- `memory_depth=1`, `num_rounds=100` (`experiments.py:130-131`)
- defaults in suite call: `generations=300` (`experiments.py:727`)
- `use_tournament_fitness=True`, `variance_penalty=0.5`, `coevolution=True`, `coevolution_k=5` (`experiments.py:104-107`)

Outputs:
- Table rows: method, hyperparameters, best fitness, runtime, best bitstring.

### 3.2 Memory-depth experiment
Source: `experiments.py:150-233`

Objective:
- Evaluate depth vs performance and complexity across all four optimizers.

Independent variable:
- `memory_depth`: `[1, 2, 3, 4, 5]` (`experiments.py:163`)

Fixed settings:
- `generations=300` in full suite (`experiments.py:735`)
- GA/EDA: `population_size=100` (`experiments.py:171`, `experiments.py:187`)
- HC: `restarts=10` (`experiments.py:202`)
- TS: `tabu_size=10` (`experiments.py:217`)

Recorded fields:
- `best_fitness`, `time_taken`, `strategy_bits` per method/depth.

### 3.3 Multi-method comparison
Source: `experiments.py:235-353`

Objective:
- Compare method stability and peak performance over multiple runs.

Run design:
- `n_runs=5` (`experiments.py:237`, `experiments.py:748`)
- `generations=500` (`experiments.py:747`)
- per-run rounds sampled in `[80, 120]` (`experiments.py:240`, `experiments.py:261-263`)
- base seed `42` (`experiments.py:239`, `experiments.py:717`)

Per-run method settings:
- GA: `population_size=100`, `mutation_rate=0.01` (`experiments.py:270`)
- EDA: `population_size=100` (`experiments.py:292`)
- HC: `restarts=10` (`experiments.py:314`)
- TS: `tabu_size=10` (`experiments.py:334`)

Outputs:
- `method_comparison.csv` and convergence histories (`experiments.py:352`, `experiments.py:757`).

### 3.4 Evolved-vs-reference tournament
Source: `experiments.py:355-455`

Objective:
- Produce one strongest evolved champion and rank against references.

Protocol details:
1. Build per-method optimizer templates (`experiments.py:372-403`).
2. Evolve one candidate per method (`experiments.py:405-413`).
3. Score candidates with uniform evaluator and choose max fitness (`experiments.py:416-424`).
4. Rename champion with method + prefix for reporting (`experiments.py:430-433`).
5. Run tournament against `REFERENCE_STRATEGIES` with `evaluation_rounds=200` (`experiments.py:370`, `experiments.py:442`).

Notable settings:
- Tournament fitness enabled for all methods (`experiments.py:377`, `experiments.py:384`, `experiments.py:392`, `experiments.py:399`).
- Coevolution disabled in this stage for GA/EDA (`experiments.py:379`, `experiments.py:386`).

### 3.5 ML scaling experiment
Source: `experiments.py:457-499`, `ml_prediction.py:94-228`

Objective:
- Evaluate model performance across increasing training sizes.

Configured in full suite:
- `train_sizes=[500, 1000, 2000]`, `memory_depth=2` (`experiments.py:773`).

Labeling rule:
- Positive label = top 20% fitness percentile (`ml_prediction.py:99`, `ml_prediction.py:135`).

Models trained:
- RandomForest, LogisticRegression, SVM, NeuralNetwork, GradientBoosting (`ml_prediction.py:180-186`).

Metrics logged:
- Accuracy, Precision, Recall, F1.

## 4. Reproducibility Workflow

Seed controls:
- `_reset_random_state(seed)` sets both Python and NumPy RNGs (`experiments.py:45-48`).
- Method-specific run seeds in `compare_all_methods` use deterministic offsets (`experiments.py:257-258`, `experiments.py:268`, `experiments.py:289`, `experiments.py:311`, `experiments.py:331`).

Metadata artifact:
- `results/reproducibility_metadata.json` written by `log_reproducibility_metadata` (`experiments.py:71-99`, `main.py:223-225`).

Observed metadata in current run:
- UTC timestamp: `2026-02-28T02:45:54Z`
- `base_seed=42`
- `seed_policy="base_seed + run index; method offsets x100 + [1..4]"`
- Python `3.11.9`; NumPy `2.4.2`; SciPy `1.17.1`; sklearn `1.8.0`.

## 5. Artifact Schema Reference (CSV/JSON)

### 5.1 CSV outputs

| File | Columns | Meaning |
|---|---|---|
| `ga_parameter_tuning.csv` | `method,population_size,mutation_rate,best_fitness,time_taken,best_strategy` | GA grid search outcomes |
| `memory_depth_results.csv` | `method,memory_depth,best_fitness,time_taken,strategy_bits` | Depth impact by method |
| `method_comparison.csv` | `run,method,best_fitness,time_taken,final_gen` | Multi-run method comparisons |
| `tournament_results.csv` | `strategy,avg_score,total_score,matches,type` | Champion + references tournament ranking |
| `ml_results.csv` | `train_size,model,accuracy,precision,recall,f1,memory_depth` | ML scaling performance |
| `pareto_frontier.csv` | `method,memory_depth,best_fitness,time_taken,strategy_bits,pareto_optimal` | Non-dominated complexity/fitness points |

### 5.2 JSON outputs

| File | Top-level structure | Meaning |
|---|---|---|
| `statistical_analysis.json` | object with `anova_f`, `anova_p`, `effect_sizes`, `confidence_intervals` | Statistical significance summary |
| `reproducibility_metadata.json` | object with runtime + environment fields | Run provenance metadata |
| `zd_analysis.json` | array of strategy objects (`is_zd`, `chi`, `phi`, `classification`) | ZD linearity diagnostics |

## 6. Observed Numeric Outcomes (Current `results/`)

Primary source files:
- `results/comprehensive_report.txt`
- `results/*.csv`
- `results/statistical_analysis.json`

### 6.1 GA tuning highlights
- Best observed configuration: `population_size=200`, `mutation_rate=0.05`, `best_fitness=257.71473741287303` (`ga_parameter_tuning.csv`).
- `best_strategy` for top two configs is `00110`.

### 6.2 Memory-depth highlights
- Peak fitness overall in this table: GA at depth 3 with `266.12014479031325`.
- Pareto points reported in `pareto_frontier.csv`: EDA depth 1, GA depth 2, GA depth 3.
- Complexity growth is exact: 5 -> 17 -> 65 -> 257 -> 1025 bits.

### 6.3 Method comparison highlights
- Mean best fitness:
  - EDA `251.559653`
  - GA `245.725297`
  - HillClimbing `224.215343`
  - TabuSearch `224.215343`
- In current run, HillClimbing and TabuSearch rows are numerically identical across runs.

### 6.4 Tournament highlights
- Top 3 by `avg_score`:
  1. TF2T `538.4`
  2. TFT `536.6`
  3. `EvolvedChampion_GA_00101` `534.8`

### 6.5 ML highlights
- Best aggregate F1 in report summary: NeuralNetwork `0.344255` (grouped means in `comprehensive_report.txt`).
- At train size 2000, best single-row F1 is NeuralNetwork `0.4461538461538462` (`ml_results.csv`).

### 6.6 Statistical highlights
From `statistical_analysis.json`:
- ANOVA: `F=0.651513434481137`, `p=0.5934853222459717`.
- Largest listed pairwise effect size by magnitude is EDA vs HC/TS with Cohen's d ~ `0.6633`.

### 6.7 ZD highlights
From `zd_analysis.json`:
- 4 analyzed strategies, all labeled `Generous` with `is_zd=true`.
- `chi` values are approximately `0.977` to `0.979` (< 1).

## 7. Interpretation Boundaries (Non-Overclaim Rules)

Use these constraints when writing conclusions:
1. Current method rankings are run-configuration-specific, not universal algorithm rankings.
2. High tournament scores indicate performance against the evaluated pool, not universal dominance.
3. ML performance is tied to this feature set and label definition (top-percentile threshold), not a general theory of cooperation.
4. Pareto frontier reflects current objective dimensions (`best_fitness`, `strategy_bits`) only.
5. ZD classification here is approximate linear-fit based (`zd_analysis.py:13-52`), not a formal proof of ZD strategy derivation.

## 8. Statistical Interpretation Guide

Source function:
- `analysis.statistical_significance_test` (`analysis.py:32-92`).

How to read outputs:
1. `anova_p` assesses whether mean differences across methods are statistically distinguishable in this sample.
2. `effect_sizes` gives pairwise magnitude context even when p-values are not significant.
3. `confidence_intervals` reflect uncertainty of per-method means over 5 runs.

Current run interpretation:
- ANOVA p-value is > 0.05, so no strong evidence of method mean separation under this run design.
- Effect sizes suggest practical separation between EDA and HC/TS may still exist and should be re-tested with higher run counts.

## 9. Pareto and ZD Interpretation Anchors

### Pareto
- Generation path: `main.py:196-201` + `analysis.py:95-126`.
- Practical reading: the frontier points are best trade-offs between complexity (`strategy_bits`) and fitness.
- In current run, frontier ends at GA depth 3; deeper memories were dominated in this 2D objective view.

### ZD
- Generation path: `main.py:203-215` + `zd_analysis.py:57-86`.
- Practical reading: evolved strategy set in this run appears as generous under fitted `chi < 1`.
- Use wording "classified as approximately generous under this diagnostic" instead of deterministic game-theory proof.

## 10. Reconciliation Notes for Reporting

1. `comprehensive_report.txt` includes `convergence_comparison.png` in generated files, while full-run pipeline currently emits `convergence_ci.png` (`main.py:220`, `experiments.py:619`).
2. ML summary narrative in section 4.4 of `comprehensive_report.txt` states Random Forest and Gradient Boosting perform best, but section 3.5 table and "Best ML Model" line show NeuralNetwork highest mean F1.
3. Documentation and paper text should prioritize artifact-backed values over static narrative boilerplate.

## 11. Experiment Documentation QA Checklist

- Every experiment objective is linked to a concrete function anchor.
- Every CSV/JSON output has explicit schema and interpretation context.
- All quoted numbers are traceable to current files in `results/`.
- Interpretation boundaries are present and avoid unsupported generalizations.
- No standalone risk appendix label is used; interpretation limits are embedded in protocol/result sections.
