# Architecture Reference

This document defines how the repository is structured, how runtime modes execute, and how data/control move across modules.

Related docs:
- [DOCS_INDEX.md](./DOCS_INDEX.md)
- [INTERFACES_AND_DATA_CONTRACTS.md](./INTERFACES_AND_DATA_CONTRACTS.md)
- [ALGORITHMS_DEEP_DIVE.md](./ALGORITHMS_DEEP_DIVE.md)
- [EXPERIMENTS.md](./EXPERIMENTS.md)
- [TRACEABILITY_MATRIX.md](./TRACEABILITY_MATRIX.md)

## 1. System Objectives and Architecture Principles

Primary system objective:
- Evolve and evaluate IPD strategies using multiple optimization algorithms, then analyze performance statistically and through ML-assisted pattern modeling.

Core principles implemented in code:
1. Separation of concerns:
   - Game rules and strategy encoding are isolated in `ipd_core.py:26-259`.
   - Search methods and evaluation are isolated in `optimization.py:40-759`.
   - Experiment orchestration is centralized in `experiments.py:33-793`.
2. Reproducibility-aware orchestration:
   - Controlled seeding in `experiments.py:45-48`, `experiments.py:257-263`, `experiments.py:717-720`.
   - Runtime metadata logging in `experiments.py:71-99`.
3. Evidence-first outputs:
   - `--full` pipeline writes tabular and visual artifacts used by the report in `main.py:169-226`.
4. Performance-aware evaluation:
   - Global LRU-like caches and locking in `optimization.py:44-84`, `optimization.py:92-104`, `optimization.py:120-148`.

## 2. Runtime Modes and Entry Points

CLI interface:
- Argument parser in `main.py:425-434`.
- Dispatch logic in `main.py:436-442`.

| Mode | Command | Entry function | Purpose | Primary outputs |
|---|---|---|---|---|
| Demo | `python main.py --demo` (or no flag) | `run_quick_demo()` (`main.py:50`) | Fast showcase of game, optimization, and ML subsystems | Console-only |
| Full | `python main.py --full` | `run_full_experiments()` (`main.py:169`) | Full experiment generation and post-analysis | `results/*.csv`, `results/*.png`, `results/*.json`, `results/comprehensive_report.txt` |
| Report | `python main.py --report` | `generate_comprehensive_report()` (`main.py:232`) | Rebuild textual report from existing `results/` tables | `results/comprehensive_report.txt` |

## 3. End-to-End Execution Flows

### 3.1 Demo flow
1. Print payoff matrix and run direct matches (`main.py:55-74`).
2. Run small tournament with reference strategies via `IPDGame.round_robin_tournament()` (`main.py:76-88`, `ipd_core.py:88-121`).
3. Run GA, EDA, HillClimbing, TabuSearch with low iteration budgets (`main.py:90-132`).
4. Generate ML data and train five classifiers (`main.py:137-157`, `ml_prediction.py:94-138`, `ml_prediction.py:150-228`).

### 3.2 Full flow
1. Call experiment suite orchestrator: `run_full_experiment_suite(output_dir)` (`main.py:177`, `experiments.py:703-789`).
2. Compute and persist post-analysis:
   - ANOVA/effect sizes -> `statistical_analysis.json` (`main.py:187-194`, `analysis.py:32-92`).
   - Pareto CSV/plot -> `pareto_frontier.csv` and `pareto_frontier.png` (`main.py:196-201`, `analysis.py:95-126`, `experiments.py:625-656`).
   - ZD diagnostics -> `zd_analysis.json` (`main.py:203-215`, `zd_analysis.py:13-86`).
   - Convergence CI -> `convergence_ci.png` (`main.py:218-220`, `experiments.py:658-701`).
   - Metadata -> `reproducibility_metadata.json` (`main.py:223-225`, `experiments.py:71-99`).
3. Generate final textual report from CSVs (`main.py:228`, `main.py:232-420`).

### 3.3 Report-only flow
1. Read existing CSVs from `results/`.
2. Compose summary text and persist `results/comprehensive_report.txt` (`main.py:235-420`).

## 4. Module Boundaries and Dependencies

### 4.1 Dependency graph (text form)
- `ipd_core.py` -> no internal dependency on project modules.
- `optimization.py` -> depends on `ipd_core.py`.
- `ml_prediction.py` -> depends on `ipd_core.py` and `optimization.py`.
- `analysis.py` -> depends on `ipd_core.py` and `optimization.py`.
- `zd_analysis.py` -> depends on `ipd_core.py`.
- `experiments.py` -> depends on `ipd_core.py`, `optimization.py`, `ml_prediction.py`, and plotting/stat libs.
- `main.py` -> top-level composition layer, imports all modules above.

### 4.2 Component responsibilities

#### `ipd_core.py`
- Payoff matrix and game engine (`PAYOFF_MATRIX`, `IPDGame`) at `ipd_core.py:26-123`.
- Strategy abstraction and identity semantics at `ipd_core.py:34-52`.
- Baseline strategy library at `ipd_core.py:129-207`.
- Bitstring strategy compiler at `ipd_core.py:210-257`.

#### `optimization.py`
- Shared evaluation core (`FitnessEvaluator`) at `optimization.py:40-170`.
- GA implementation at `optimization.py:175-360`.
- EDA implementation at `optimization.py:363-495`.
- HillClimbing implementation at `optimization.py:498-620`.
- TabuSearch implementation at `optimization.py:623-756`.

#### `ml_prediction.py`
- Behavioral feature extraction at `ml_prediction.py:41-91`.
- Dataset generation and labeling at `ml_prediction.py:94-138`.
- Multi-model training/prediction/search at `ml_prediction.py:141-268`.

#### `experiments.py`
- Canonical opponent policy at `experiments.py:33`, `experiments.py:50-60`.
- Experiment runners at `experiments.py:101-499`.
- Visualization layer at `experiments.py:503-701`.
- Full suite orchestration at `experiments.py:703-789`.

#### `analysis.py` and `zd_analysis.py`
- Statistical tests + Pareto extraction in `analysis.py:32-126`.
- ZD linearity approximation and classification in `zd_analysis.py:13-86`.

#### `main.py`
- CLI, orchestration, post-processing, and report writer at `main.py:50-442`.

## 5. Data and Control Contracts Between Layers

| Producer | Artifact / object | Consumer | Contract notes |
|---|---|---|---|
| `ipd_core.create_strategy_from_bitstring` (`ipd_core.py:210`) | `Strategy` | `optimization`, `ml_prediction`, `analysis`, `experiments` | `bitstring` length must match `1 + 4^n` (`ipd_core.py:220-223`) |
| `FitnessEvaluator.evaluate(_population)` (`optimization.py:106`, `optimization.py:151`) | float fitness / list[float] | all optimizers, ML data generation | Supports pairwise and tournament-based scoring; optional variance penalty (`optimization.py:128-140`) |
| Optimizers `.evolve()` | `OptimizationResult` (`optimization.py:29-37`) | `experiments.py`, demo path | Standardized fields: best strategy, fitness history, params, runtime |
| `ExperimentRunner.*` | `pandas.DataFrame` per experiment | `main.py`, visualization functions | Persisted as CSV files under `results/` |
| `analysis` and `zd_analysis` | dict/list JSON payloads | `main.py` output writer | Stored as `statistical_analysis.json`, `zd_analysis.json`, etc. |

## 6. Strategy Encoding Specification (`1 + 4^n`)

Formal definition:
- Bit 0: initial move (`0` cooperate, `1` defect).
- Remaining bits index conditional responses from interaction history.

Implementation anchors:
- Length calculation and assertion: `ipd_core.py:220-223`.
- Depth-1 mapping rule: `(my_last, opp_last)` -> index 1..4 at `ipd_core.py:234-238`.
- Depth-n mapping rule: base-4 rolling index at `ipd_core.py:239-246`.

Worked examples:
1. Memory depth 1:
   - Length = `1 + 4^1 = 5`.
   - TFT clone bitstring is `00101` (commented in `ipd_core.py` test block).
2. Memory depth 2:
   - Length = `1 + 4^2 = 17`.
3. Memory depth 3:
   - Length = `1 + 4^3 = 65`.

Engineering consequence:
- Search space doubles per bit, so total strategies are `2^(1+4^n)`.
- This dominates runtime at higher memory depths (see `results/memory_depth_results.csv`).

## 7. Caching Subsystem Internals

Cache architecture:
- Match cache: `_GLOBAL_MATCH_CACHE` (`optimization.py:44`) keyed by `(strategy_key, opponent_key, num_rounds)` (`optimization.py:91`).
- Fitness cache: `_GLOBAL_FITNESS_CACHE` (`optimization.py:45`) keyed by strategy/opponent tuple plus evaluation flags (`optimization.py:119`).

Concurrency and eviction:
- Global lock: `_GLOBAL_LOCK` (`optimization.py:46`) used in `_cache_get` and `_cache_put` (`optimization.py:71-84`).
- Eviction policy is recency-based using `OrderedDict.move_to_end` and FIFO pop of oldest (`optimization.py:75`, `optimization.py:82-84`).
- Bounded by `_MAX_MATCH_CACHE_SIZE` and `_MAX_FITNESS_CACHE_SIZE` (both 10000; `optimization.py:42-43`).

Lifecycle:
- Caches are explicitly cleared at experiment boundaries (`experiments.py:113`, `experiments.py:161`, `experiments.py:252`, `experiments.py:366`, `experiments.py:467`).

## 8. Determinism, Randomness, and Reproducibility

Random sources:
- Python `random` drives bit generation, selection, crossover points, mutations, and neighborhood sampling.
- NumPy random may be used by libraries and calculations.

Control points:
- Seed reset helper: `_reset_random_state(seed)` in `experiments.py:45-48`.
- Run-level seed pattern: `run_seed = base_seed + run` (`experiments.py:257`), with method offsets `*100 + [1..4]` (`experiments.py:268`, `experiments.py:289`, `experiments.py:311`, `experiments.py:331`).
- Persisted metadata: `results/reproducibility_metadata.json` via `experiments.py:71-99`.

Observed metadata baseline:
- `base_seed = 42`, `cpu_count = 16`, `numpy=2.4.2`, `scipy=1.17.1`, `sklearn=1.8.0` in `results/reproducibility_metadata.json`.

## 9. Complexity and Scalability Profile

Let:
- `P` = population size.
- `G` = generations/iterations.
- `O` = number of opponents per evaluation.
- `R` = rounds per match.
- `B` = strategy bit length.

Approximate costs:
- Fitness evaluation: `O(O * R)` per strategy.
- GA/EDA (without cache hits): `O(G * P * O * R)`.
- HillClimbing: `O(iter * neighbors * O * R)`, neighbors approx `B` or sampled cap (`optimization.py:531-533`).
- TabuSearch: same neighborhood class as HC, typically `max_iter` loops.

Observed scaling evidence:
- `results/memory_depth_results.csv` shows large runtime increase from depth 1 to depth 5 for GA/EDA and especially TabuSearch.
- Bit-length progression matches `1 + 4^n` (5, 17, 65, 257, 1025).

## 10. Edge Cases, Failure Behavior, and Guardrails

1. Bitstring length mismatch:
   - Hard assertion in `ipd_core.py:221` prevents invalid policy construction.
2. Empty opponent set:
   - Fitness evaluation returns `0.0` if no scores (`optimization.py:124-125`).
3. Small population parallel overhead:
   - Parallel path disabled for populations `< 10` (`optimization.py:157-160`).
4. Deep memory neighborhood explosion:
   - HC/TS cap default neighbors to 50 when `memory_depth > 2` (`optimization.py:518`, `optimization.py:643`).
5. ML class-balance fragility:
   - Stratification only when each class has at least 2 members (`ml_prediction.py:164-166`).
6. Feature fidelity at depth > 1:
   - `extract_features` uses aggregate placeholders for conditional response fields (`ml_prediction.py:82-89`).
7. Report artifact naming mismatch:
   - Report text lists `convergence_comparison.png` (`main.py:409`), while active full flow writes `convergence_ci.png` (`main.py:220`, `experiments.py:619`).

## 11. Extension Points

### Add a new optimization method
1. Implement class with `.evolve(opponents, generations, verbose)` returning `OptimizationResult`.
2. Reuse `FitnessEvaluator` for consistent scoring.
3. Register method in:
   - `ExperimentRunner.compare_all_methods` (`experiments.py:235-353`).
   - `ExperimentRunner.evolved_vs_reference` if needed (`experiments.py:372-403`).
   - Visualization and reporting sections.

### Add a new ML model
1. Add model in `StrategyPredictor.train()` model dictionary (`ml_prediction.py:180-186`).
2. Ensure prediction path handles scaling needs (`ml_prediction.py:193-198`, `ml_prediction.py:241-242`).
3. Extend artifact interpretation docs and report summaries.

### Add new experiment artifact
1. Add experiment runner function in `experiments.py`.
2. Persist DataFrame/JSON in `run_full_experiment_suite` (`experiments.py:703-789`).
3. Register post-processing in `main.run_full_experiments` (`main.py:169-226`).
4. Add schema to [INTERFACES_AND_DATA_CONTRACTS.md](./INTERFACES_AND_DATA_CONTRACTS.md) and interpretation to [RESULTS_ATLAS.md](./RESULTS_ATLAS.md).

## 12. Architecture Checklist for Review

- All seven source modules are represented with explicit boundaries.
- Runtime modes are mapped to exact CLI paths.
- Data contracts define handoff types across layers.
- Randomness and reproducibility controls are explicitly captured.
- Performance mechanisms (cache + neighbor caps) are documented.
- Known reconciliation items are documented without overstating behavior.
