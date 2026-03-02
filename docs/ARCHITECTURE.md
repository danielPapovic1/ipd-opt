# ARCHITECTURE

## 1) System Design Summary

This project is split into five major layers:
1. **Game-theoretic core** (`ipd_core.py`)
2. **Optimization engines** (`optimization.py`)
3. **ML prediction pipeline** (`ml_prediction.py`)
4. **Experiment orchestration** (`experiments.py`)
5. **Post-analysis/statistics** (`analysis.py`, `zd_analysis.py`)

The architecture keeps simulation logic separate from optimization and analytics, so each part can evolve independently.

---

## 2) File-by-File Component Breakdown

## `ipd_core.py`
**Role:** Foundational game engine and strategy library.

**Key elements:**
- `PAYOFF_MATRIX` for standard IPD rewards.
- `Move` enum (`COOPERATE`, `DEFECT`).
- `Strategy` abstraction (`name`, `play_func`, `bitstring`, `memory_depth`).
- `IPDGame.play_match()` for repeated interaction.
- Reference strategies: `TFT`, `TF2T`, `STFT`, `ALL-D`, `ALL-C`, `RAND`, `GRIM`, `PAVLOV`.
- `create_strategy_from_bitstring()` for encoded policies.

**Design intent:** isolate pure game rules from search and ML code.

## `optimization.py`
**Role:** Evolutionary and local-search optimization.

**Key elements:**
- `FitnessEvaluator`: tournament/pairwise fitness evaluation.
- `GeneticAlgorithm`, `EDA`, `HillClimbing`, `TabuSearch`.
- `OptimizationResult` for standardized outputs.

**GA pipeline:**
Roulette-wheel selection → single-point crossover → bit-flip mutation.

**Caching subsystem (major performance feature):**
- `_GLOBAL_MATCH_CACHE` stores strategy-vs-strategy score lookups.
- `_GLOBAL_FITNESS_CACHE` stores computed fitness values over opponent sets and config.
- Shared locking + bounded cache sizes avoid unbounded growth.

## `ml_prediction.py`
**Role:** Supervised learning over strategy-derived features.

**Key elements:**
- `extract_features()` builds a compact feature vector from strategy behavior.
- `generate_training_data()` samples random strategies and labels top-percentile performers.
- `StrategyPredictor.train()` trains five models:
  - RandomForest
  - LogisticRegression
  - SVM
  - NeuralNetwork (MLP)
  - GradientBoosting

**Feature set (documented behavior):**
- `initial_cooperate`
- `coop_rate`
- `response_to_cc`
- `response_to_cd`
- `response_to_dc`
- `response_to_dd`
- `nice`
- `provokable`
- `forgiving`
- `tft_like`

## `experiments.py`
**Role:** Coordinated experiment runner + artifact generation.

**Key elements:**
- `ExperimentRunner.run_parameter_tuning_ga()`
- `ExperimentRunner.run_memory_depth_experiment()`
- `ExperimentRunner.compare_all_methods()`
- `ExperimentRunner.evolved_vs_reference()`
- `ExperimentRunner.run_ml_experiments()`

**Reproducibility controls:**
- `_reset_random_state(seed)` for deterministic runs.
- structured run seeding in comparative loops.
- metadata logging with runtime/library info.

## `analysis.py` and `zd_analysis.py`
**Role:** Statistical validation and strategy-behavior diagnostics.

**Key elements:**
- `statistical_significance_test()` (ANOVA + pairwise comparisons + Cohen’s d).
- `pareto_frontier_analysis()` to identify non-dominated complexity/fitness trade-offs.
- `is_zero_determinant()` and `analyze_zd_properties()` for approximate ZD classification.

---

## 3) Strategy Encoding: `1 + 4^n`

Bitstring strategies encode deterministic responses conditioned on history.

- For `memory_depth = 1`, encoding is 5 bits:
  - bit 0 = initial action
  - bits 1..4 map to `(C,C), (C,D), (D,C), (D,D)`
- For general depth `n`, length is:
  - **`1 + 4^n`**

The implementation computes this directly in `create_strategy_from_bitstring()` and in optimizer bit-length setup logic.

---

## 4) Data and Control Flow

1. Optimizers generate candidate bitstrings.
2. Candidates are converted into `Strategy` instances.
3. `FitnessEvaluator` simulates against opponents via `IPDGame`.
4. Caches short-circuit repeated evaluations.
5. Experiments aggregate results into dataframes and plots.
6. Analysis computes significance tests, Pareto frontiers, and ZD diagnostics.
