# Algorithms Deep Dive

This document explains how each optimization method is implemented in this repository, how objectives are computed, and how parameter choices influence outcomes.

Related docs:
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [EXPERIMENTS.md](./EXPERIMENTS.md)
- [RESULTS_ATLAS.md](./RESULTS_ATLAS.md)

## 1. Shared Objective Layer: `FitnessEvaluator`

Implementation: `optimization.py:40-170`.

Core job:
- Convert a candidate `Strategy` into a scalar fitness against a selected opponent pool.

Evaluation modes:
1. Pairwise average mode (default):
   - Fitness = mean score across opponents (`optimization.py:134`).
2. Tournament mode:
   - If `use_tournament_fitness=True` and not coevolution, evaluate in no-self-play round robin (`optimization.py:128-133`).
3. Variance-penalized mode:
   - Fitness = base - `variance_penalty * std(scores)` (`optimization.py:137-140`).

Caching integration:
- Match-level cache and fitness-level cache reduce repeated simulations (`optimization.py:44-45`, `optimization.py:91-104`, `optimization.py:119-148`).

Coevolution option:
- Population members can be evaluated partly against sampled peers (`optimization.py:165-169`).

## 2. Genetic Algorithm (GA)

Implementation: `optimization.py:175-360`.

## 2.1 Pipeline
1. Initialize random population (`optimization.py:217-220`).
2. Score population with evaluator (`optimization.py:287-290`).
3. Per generation:
   - Keep elites (`optimization.py:313-316`).
   - Select parents via roulette wheel (`optimization.py:222-238`).
   - Single-point crossover (`optimization.py:240-254`).
   - Bit-flip mutation (`optimization.py:256-264`).
   - Re-evaluate (`optimization.py:335-338`).
4. Return best-ever strategy and histories.

## 2.2 Important implementation details
- Roulette wheel shifts fitness values to non-negative range (`optimization.py:225-227`).
- Crossover uses one random cut in `[1, len(bits)-1]` (`optimization.py:246`).
- Mutation scans all bits independently with Bernoulli probability `mut_rate`.
- Elitism default is 2 (`optimization.py:191`).

## 2.3 Defaults and experiment overrides
Default constructor values:
- `population_size=100`, `mutation_rate=0.01`, `crossover_rate=0.8`, `elitism=2` (`optimization.py:188-191`).

Experiment overrides:
- GA tuning grid modifies pop size and mutation (`experiments.py:116-117`).
- Evolved-vs-reference increases pop size to 200 (`experiments.py:374`).

## 3. Estimation of Distribution Algorithm (EDA)

Implementation: `optimization.py:363-495`.

## 3.1 Pipeline
1. Initialize probability vector at 0.5 for all bits (`optimization.py:433`).
2. Sample bitstrings from Bernoulli vector (`optimization.py:395-402`).
3. Evaluate fitness.
4. Select top fraction (`selection_rate`) (`optimization.py:460-463`).
5. Update probability vector with smoothing via `learning_rate` (`optimization.py:404-416`).
6. Resample and iterate.

## 3.2 Behavior implications
- EDA performs distribution-level search instead of explicit crossover.
- Smooth updates reduce abrupt policy collapse when `learning_rate` is moderate.

## 3.3 Defaults and experiment overrides
Defaults:
- `population_size=100`, `selection_rate=0.3`, `learning_rate=0.1` (`optimization.py:371-373`).

Experiment usage:
- Standard method comparison uses defaults (`experiments.py:292-296`).
- Tournament-champion stage sets pop size to 200 (`experiments.py:382`).

## 4. Hill Climbing (HC)

Implementation: `optimization.py:498-620`.

## 4.1 Pipeline
1. For each restart (`restarts`):
   - Random initial strategy (`optimization.py:573`).
   - Generate single-bit-flip neighbors (`optimization.py:525-539`).
   - Move if best neighbor strictly improves (`optimization.py:585-590`).
2. Track best-overall across restarts.

## 4.2 Neighborhood control
- Full neighborhood is all bit positions.
- For deep memory (`memory_depth > 2`) and unspecified `max_neighbors`, sample cap = 50 (`optimization.py:518`).

## 4.3 Iteration budget behavior
- Effective local budget per restart: `max_iter // restarts` (`optimization.py:579`).
- `generations` argument is accepted and mapped to `max_iterations` for API compatibility (`optimization.py:548-550`).

## 5. Tabu Search (TS)

Implementation: `optimization.py:623-756`.

## 5.1 Pipeline
1. Start from random strategy (`optimization.py:689`).
2. At each iteration:
   - Generate neighbors (`optimization.py:650-664`).
   - Evaluate all neighbors (`optimization.py:705-706`).
   - Choose best non-tabu candidate, with aspiration if tabu candidate beats best-ever (`optimization.py:709-720`).
3. Update tabu list with bounded size (`optimization.py:730-732`).
4. Track best-ever solution.

## 5.2 Key controls
- `tabu_size` default 10 (`optimization.py:630`).
- Same deep-memory neighbor capping rule as HC (`optimization.py:643`, `optimization.py:656-657`).

## 6. Objective Variants in Experiments

From `experiments.py`:

1. GA tuning and method comparison typically use:
   - `use_tournament_fitness=True`
   - `variance_penalty=0.5`
   - Coevolution enabled for GA/EDA in these experiment families (`experiments.py:105-107`, `experiments.py:242-244`).
2. Evolved-vs-reference stage uses:
   - Tournament fitness true.
   - Lower variance penalty (`0.25`) for candidate evolution (`experiments.py:378`, `experiments.py:385`, `experiments.py:393`, `experiments.py:400`).
   - Final champion scoring with variance penalty `0.0` (`experiments.py:420`).

Implication:
- Fitness values across experiments are comparable only when objective settings are aligned; this is one reason experiments are interpreted by family.

## 7. Comparative Tradeoffs in Current Results

Artifact evidence:
- `results/method_comparison.csv`
- `results/memory_depth_results.csv`
- `results/comprehensive_report.txt`

Observed patterns in this run:
1. EDA has highest mean best-fitness in method comparison (251.559653).
2. GA is close behind and reaches strong peaks.
3. HC and TS are much faster at low depth but weaker in mean best-fitness.
4. Runtime growth with depth is steep for all methods; TS becomes especially expensive at depth 5.

## 8. Tuning Guidance (Repository-Specific)

## 8.1 GA
- Increase `population_size` when search space grows (higher memory depth).
- Use moderate-to-high mutation when convergence stalls; in current GA tuning, 0.05 with larger populations produced top results.
- Keep elitism > 0 to preserve best candidates.

## 8.2 EDA
- If convergence is unstable, lower `learning_rate` to smooth vector updates.
- If progress is slow, raise `selection_rate` to increase pressure.

## 8.3 HC and TS
- For high depths, avoid full neighborhoods unless runtime budget is large.
- Increase `restarts` (HC) for broader basin exploration.
- Increase `tabu_size` (TS) to reduce immediate cycling in rugged landscapes.

## 8.4 Cross-method comparisons
- Keep opponent pool, rounds, and objective flags fixed for fair comparison.
- Use multiple seeds and report confidence intervals, not only single-run maxima.

## 9. Implementation Risks and Guardrails

1. Name-based `Strategy` equality can collapse distinct strategies if names collide (`ipd_core.py:46-51`).
2. Deep-memory strategies produce very large bitstrings; neighborhood-based methods may require explicit sampling caps.
3. ML-guided search quality is sensitive to feature fidelity at depth > 1 because response-specific features are aggregated (`ml_prediction.py:82-89`).
4. Report narratives should be reconciled with numeric artifacts for consistency.

## 10. Quick Verification Anchors

- GA methods: `optimization.py:217-264`, `optimization.py:266-360`.
- EDA methods: `optimization.py:395-416`, `optimization.py:418-495`.
- HC methods: `optimization.py:525-539`, `optimization.py:542-620`.
- TS methods: `optimization.py:650-664`, `optimization.py:667-756`.
- Shared objective and caching: `optimization.py:40-170`.
