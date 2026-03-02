# Reproducibility and Provenance

This document explains how run reproducibility is controlled, what is deterministic vs variable, and how each artifact is produced from code paths.

Related docs:
- [EXPERIMENTS.md](./EXPERIMENTS.md)
- [RESULTS_ATLAS.md](./RESULTS_ATLAS.md)
- [TRACEABILITY_MATRIX.md](./TRACEABILITY_MATRIX.md)

## 1. Reproducibility Metadata Baseline

Metadata file:
- `results/reproducibility_metadata.json`

Current recorded values:
- `timestamp_utc`: `2026-02-28T02:45:54Z`
- `python_version`: `3.11.9`
- `numpy_version`: `2.4.2`
- `scipy_version`: `1.17.1`
- `sklearn_version`: `1.8.0`
- `cpu_count`: `16`
- `base_seed`: `42`
- `seed_policy`: `base_seed + run index; method offsets x100 + [1..4]`
- `total_runtime_hours`: `0.0021208260456720986`

Generation path:
- `ExperimentRunner.log_reproducibility_metadata` (`experiments.py:71-99`)
- called from `main.run_full_experiments` (`main.py:223-225`)

## 2. Seed and Run-Policy Traceability

## 2.1 Seed control entrypoint
- `_reset_random_state(seed)` calls `random.seed(seed)` and `np.random.seed(seed)` (`experiments.py:45-48`).

## 2.2 Method comparison seed schedule
Inside `compare_all_methods` (`experiments.py:235-353`):
1. Run seed: `base_seed + run` (`experiments.py:257`).
2. GA seed: `run_seed * 100 + 1` (`experiments.py:268`).
3. EDA seed: `run_seed * 100 + 2` (`experiments.py:289`).
4. HC seed: `run_seed * 100 + 3` (`experiments.py:311`).
5. TS seed: `run_seed * 100 + 4` (`experiments.py:331`).

## 2.3 Additional controlled randomness
- Rounds per run sampled in `(80, 120)` with seeded RNG (`experiments.py:240`, `experiments.py:261-263`).
- Random strategy generation and optimizer operators use Python RNG.

## 3. Deterministic vs Variable Outputs

## 3.1 Deterministic under fixed code + dependencies + seed policy
Expected to be reproducible to very close values:
- CSV row schema and ordering.
- Tournament ranking format.
- JSON key structures.
- Plot filenames and types.

## 3.2 Potentially variable across environments
Possible differences due to library/platform behavior:
1. Floating-point tie-break behavior and tiny numeric drift.
2. Thread scheduling effects in parallel evaluation paths.
3. sklearn model fitting details if library version changes.
4. Runtime durations (`time_taken`) and derived runtime metadata.

## 3.3 Practical classification by artifact

| Artifact | Reproducibility class | Notes |
|---|---|---|
| `ga_parameter_tuning.csv` | High with fixed environment | Sensitive to RNG and objective flags |
| `memory_depth_results.csv` | High with fixed environment | Runtime columns hardware-sensitive |
| `method_comparison.csv` | Medium-high | includes random rounds in [80,120] but seeded |
| `tournament_results.csv` | High | deterministic given selected champion strategies |
| `ml_results.csv` | Medium-high | model training uses fixed random_state for models and split |
| `statistical_analysis.json` | High | deterministic function of method comparison table |
| `pareto_frontier.csv` | High | deterministic function of memory-depth table |
| `zd_analysis.json` | High | deterministic given input strategies and opponents |
| `reproducibility_metadata.json` | Medium | timestamp and runtime expected to differ |
| PNG plots | High shape, medium pixels | rendering backend/version can alter anti-aliasing |

## 4. Provenance Chain (Function -> Artifact)

| Artifact | Producer function(s) | Invocation path |
|---|---|---|
| `ga_parameter_tuning.csv` | `ExperimentRunner.run_parameter_tuning_ga` (`experiments.py:101`) | `run_full_experiment_suite` (`experiments.py:727-731`) |
| `memory_depth_results.csv` | `ExperimentRunner.run_memory_depth_experiment` (`experiments.py:150`) | `run_full_experiment_suite` (`experiments.py:735-739`) |
| `method_comparison.csv` | `ExperimentRunner.compare_all_methods` (`experiments.py:235`) | `run_full_experiment_suite` (`experiments.py:744-756`) |
| `tournament_results.csv` | `ExperimentRunner.evolved_vs_reference` (`experiments.py:355`) | `run_full_experiment_suite` (`experiments.py:761-766`) |
| `ml_results.csv` | `ExperimentRunner.run_ml_experiments` (`experiments.py:457`) | `run_full_experiment_suite` (`experiments.py:773-777`) |
| `ga_parameter_heatmap.png` | `create_visualizations` (`experiments.py:503`) | `run_full_experiment_suite` (`experiments.py:782`) |
| `memory_depth_impact.png` | `create_visualizations` | same |
| `method_comparison.png` | `create_visualizations` | same |
| `ml_comparison.png` | `create_visualizations` | same |
| `tournament_results.png` | `create_visualizations` | same |
| `evolved_comparison.png` | `create_visualizations` | same |
| `pareto_frontier.png` | `plot_pareto_frontier` (`experiments.py:625`) | via visualization + post-processing |
| `convergence_ci.png` | `plot_convergence_ci` (`experiments.py:658`) | via visualization and `main.py:218-220` |
| `statistical_analysis.json` | `statistical_significance_test` (`analysis.py:32`) | `main.py:187-194` |
| `pareto_frontier.csv` | `pareto_frontier_analysis` (`analysis.py:95`) | `main.py:196-201` |
| `zd_analysis.json` | `analyze_zd_properties` (`zd_analysis.py:57`) | `main.py:203-215` |
| `reproducibility_metadata.json` | `log_reproducibility_metadata` (`experiments.py:71`) | `main.py:223-225` |
| `comprehensive_report.txt` | `generate_comprehensive_report` (`main.py:232`) | `main.py:228`, `main.py:438-440` |

## 5. Report-Level Provenance Notes

`results/comprehensive_report.txt` is generated from persisted tables by `main.generate_comprehensive_report` (`main.py:232-420`).

Important implication:
- Narrative lines may lag behind numeric table truth if boilerplate text is not updated.
- Paper writing should prioritize numeric artifacts and use report text as a synthesis layer.

## 6. Verification Checklist Without Re-running Experiments

1. Confirm all expected files exist in `results/`.
2. Verify CSV headers match schema documented in [INTERFACES_AND_DATA_CONTRACTS.md](./INTERFACES_AND_DATA_CONTRACTS.md).
3. Compare summary values in report section 3 with raw CSV calculations.
4. Verify statistical JSON keys: `anova_f`, `anova_p`, `effect_sizes`, `confidence_intervals`.
5. Verify metadata seed policy string matches code policy in `experiments.py:717-720` and `main.py:183-185`.
6. Confirm ZD classifications and `chi` values align with `zd_analysis.json` rows.

## 7. Reproducibility-Focused Citation Snippets

Use these references in technical writing:
- "Seed policy and metadata logging were implemented in `experiments.py:45-48` and `experiments.py:71-99`."
- "Full-run post-processing wrote statistical, Pareto, ZD, CI, and metadata artifacts in `main.py:187-225`."
- "The recorded runtime environment is documented in `results/reproducibility_metadata.json`."
