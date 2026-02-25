# Project Overview: IPD Strategy Optimization (Bitstring, Memory-1 to Memory-5) ## Goal
Implement and compare **four optimization algorithms** to discover strong **Iterated Prisoner’s Dilemma (IPD)** strategies, using a **bitstring encoding** of deterministic strategies with **memory depth m = 1..5**, and benchmark against **human-designed strategies** (at minimum: **TFT, TF2T, STFT**). **Chosen optimizers (the 4):**
1. Genetic Algorithm (GA)
2. Estimation of Distribution Algorithm (EDA)
3. Hill Climbing with Random Restarts (HC-RR)
4. Tabu Search ## What you are building (high level deliverables)
1. **IPD engine** that can run repeated matches and compute payoffs.
2. **Strategy representation**: deterministic memory-m strategy encoded as a **bitstring**.
3. **Tournament evaluator** (fitness function) that scores a candidate strategy against an opponent set.
4. **Four optimizers** that search the bitstring space.
5. **Experiment runner**: parameter sweeps + memory-depth sweeps + logging.
6. **Baselines and comparisons**: TFT, TF2T, STFT (plus a few extra simple baselines like ALLC, ALLD).
7. **Optional ML predictor**: learn a model that predicts “good strategy” from the bitstring (or derived features), and compare ML methods.
8. **Analysis + patterns**: interpret what the best strategies are doing (rules, forgiveness, retaliation, etc.). --- ## Step 0: Lock the rules of the game (do this once)
To compare methods fairly, you must freeze these choices and use them everywhere. ### 0.1 Payoff matrix
Use the standard PD ordering: **T > R > P > S**.
Pick explicit numeric values and keep them fixed in all experiments. Common default used in many IPD studies: (T, R, P, S) = (5, 3, 1, 0).
If your course materials specify something else, use that instead. ### 0.2 Horizon (number of rounds)
Pick one primary horizon and (optionally) one sensitivity test.
- Primary: fixed **N rounds per match** (example: 100 or 200).
- Sensitivity: repeat key results with a different N (example: 200 if you used 100). ### 0.3 Noise model (optional but recommended for realism)
If you include noise, define it clearly:
- **Implementation noise ε**: with probability ε, flip the chosen action (C becomes D, D becomes C).
Run with ε = 0 for your main baseline, and optionally ε ∈ {0.01, 0.05} as robustness tests. ### 0.4 Fitness scoring
Define exactly how you score strategies in tournaments:
- Average payoff per round, or total payoff over the match.
- Round-robin: candidate vs each opponent in the set, then average.
- Decide whether to include **self-play** (candidate vs itself). If you include it, include it consistently. ### 0.5 Random seeds and repeatability
Because optimization and noise are stochastic:
- Fix a list of seeds for experiments.
- Report mean and variance across seeds (at least 5 runs per setting if feasible). --- ## Step 1: Strategy representation (bitstrings) and memory depth ### 1.1 Deterministic memory-m strategies (what “memory depth” means here)
A deterministic memory-m strategy chooses C or D based on the **last m outcomes**.
Each outcome is one of: **CC, CD, DC, DD** (your action first, opponent action second). So the number of distinct histories is:
- **states(m) = 4^m**
- One bit per state decides C (0) or D (1) (or vice versa, just be consistent). ### 1.2 Canonical ordering of outcomes
Pick a canonical order for a single-round outcome, for example:
0: CC
1: CD
2: DC
3: DD A length-m history is then a base-4 number with m digits. Example for m=3:
history = (CC, DD, CD) maps to digits (0, 3, 1) -> index = 0*4^2 + 3*4^1 + 1*4^0 = 13. This gives you an index in [0, 4^m - 1] that selects the corresponding bit from the bitstring. ### 1.3 Bitstring lengths for memory depth 1 to 5
| Memory depth m | #states = 4^m | Bitstring length |
|---:|---:|---:|
| 1 | 4 | 4 bits |
| 2 | 16 | 16 bits |
| 3 | 64 | 64 bits |
| 4 | 256 | 256 bits |
| 5 | 1024 | 1024 bits | ### 1.4 Startup problem (first rounds)
Memory-m strategies need m previous outcomes, but round 1 has no history. Pick one policy and stick to it:
- **Simple fixed bootstrap (recommended for this project):** - Assume an initial history of all CC outcomes for missing rounds. - This is equivalent to “start friendly” and avoids expanding the genome.
- Alternative: encode extra “initial move” bits (more complex). Only do this if you need it. Document which bootstrap you use in the report. --- ## Step 2: Baseline opponents (human-designed strategies)
You must “compete against human designed strategies” and compare to **TFT, TF2T, STFT** at minimum. Include this baseline set in your evaluator:
- **TFT** (Tit For Tat): start C, then copy opponent’s last move.
- **TF2T** (Tit For Two Tats): start C,C; defect only if opponent defected twice in a row.
- **STFT** (Suspicious TFT): start D, then copy opponent’s last move.
- **ALLC** (Always cooperate)
- **ALLD** (Always defect)
- Optional helpful extras (good for richer evaluation): - **WSLS / Pavlov** (Win-Stay Lose-Shift) - **Random(p)** (cooperate with fixed probability p) Keep this opponent pool fixed for all experiments (or define a train pool vs test pool, see ML section). --- ## Step 3: Fitness evaluation protocol (the core bottleneck)
Every optimizer needs the same fitness signal. ### 3.1 Candidate evaluation
For a candidate bitstring strategy:
1. For each opponent in the pool: - play N rounds of IPD (with your bootstrap and optional noise) - compute candidate’s average payoff per round
2. Fitness = average across opponents (and optionally self-play). ### 3.2 Efficiency notes (this matters)
Evaluation cost dominates runtime, especially at m=4 and m=5.
Do these from day one:
- Cache pairwise match results when possible.
- Use vectorized loops or compiled loops where feasible.
- Parallelize over opponents, candidates, or seeds. --- ## Step 4: The four optimizers (what each one needs) ### 4.1 Genetic Algorithm (GA)
**Representation:** bitstring length 4^m
**Variation operators:** crossover + mutation Minimum implementation details:
- Initialize population randomly.
- Evaluate fitness for each individual.
- Selection: tournament selection or roulette wheel.
- Crossover: 1-point or 2-point crossover.
- Mutation: flip each bit with probability pm.
- Elitism: keep top k individuals each generation. Parameter sweep (start small, then expand):
- Population size: {50, 100, 200}
- Mutation rate pm: {0.001, 0.005, 0.01, 0.02}
- Crossover rate pc: {0.6, 0.8, 0.9}
- Generations: fixed evaluation budget (see Step 5) ### 4.2 Estimation of Distribution Algorithm (EDA)
**Representation:** bitstring length 4^m
**Core idea:** learn a probabilistic model of good bitstrings, sample new ones Minimum implementation details:
- Start with uniform bit probabilities (0.5 each).
- Each iteration: - sample population from current probabilities - evaluate fitness - select top fraction (elite set) - update probabilities toward elite bit frequencies - apply smoothing to avoid probabilities becoming 0 or 1 too early Parameter sweep:
- Sample size per iteration: {50, 100, 200}
- Elite fraction: {0.1, 0.2, 0.3}
- Smoothing / learning rate: {0.1, 0.2, 0.3}
- Optional: probability bounds (clip to [0.02, 0.98]) ### 4.3 Hill Climbing with Random Restarts (HC-RR)
**Representation:** bitstring length 4^m
**Neighbor move:** flip 1 bit (or k bits) Minimum implementation details:
- Start from a random bitstring.
- Repeatedly: - generate neighbor(s) by flipping bits - move to best improving neighbor (or first improving) - stop when no improvement
- Restart from a new random point and keep best found. Parameter sweep:
- Restarts: {20, 50, 100}
- Neighborhood: flip-1 vs flip-k (k in {2, 4, 8})
- Per-step candidates checked: {50, 200, 1000} (sampled neighbors) ### 4.4 Tabu Search
**Representation:** bitstring length 4^m
**Core idea:** local search plus a short-term memory (tabu list) to avoid cycling Minimum implementation details:
- Start from a random bitstring.
- At each step: - examine a set of neighbor moves (bit flips) - choose best move even if it worsens fitness - mark that move/tabu feature as tabu for L steps - allow aspiration: if a tabu move produces a new global best, allow it Parameter sweep:
- Tabu tenure L: {5, 10, 20, 50}
- Neighborhood sample size per step: {100, 500, 2000}
- Steps per run: fixed evaluation budget --- ## Step 5: Memory-depth experiment plan (m = 1..5) --- ## Step 5: Memory-depth experiment plan (m = 1..5) ### 5.1 Why you must do all depths
The project explicitly asks you to try deeper memory: **3, 4, 5**, and you also want **1 and 2** as “ramp-up” and comparison points. ### 5.2 Practical evaluation budgets (recommended)
To compare optimizers fairly, compare by **number of fitness evaluations**, not wall-clock time. Suggested evaluation budgets per run:
- m=1: 20,000 evaluations
- m=2: 30,000 evaluations
- m=3: 50,000 evaluations
- m=4: 80,000 evaluations
- m=5: 120,000 evaluations If runtime is too high, reduce budgets for m=4 and m=5 and document it. ### 5.3 Execution order (do not start at m=5)
1. Implement everything for **m=1** until the pipeline is stable.
2. Scale to **m=2** (same code path, bigger bitstring).
3. Run your full optimizer comparison at **m=3** (this is the classic size).
4. Then push to **m=4** and **m=5**. ### 5.4 What to record for every run (non-negotiable)
For each (optimizer, params, m, seed):
- best fitness found
- best bitstring strategy
- average fitness over time (learning curve)
- number of evaluations used
- runtime
- performance vs each baseline opponent (TFT, TF2T, STFT, etc.)
- robustness results if you include noise --- ## Step 6: ML prediction component (optional but aligns with project)
Purpose: build a model that predicts if a bitstring strategy will score well, so you can:
- analyze what makes strategies strong
- optionally speed up search (surrogate filtering) ### 6.1 Dataset construction
For each memory depth m:
- Generate many strategies (bitstrings) from: - uniform random sampling - snapshots from optimizer populations (GA/EDA) - local-search solutions (HC/Tabu bests)
- Label each strategy with its **true tournament fitness** under your fixed evaluation protocol. Recommended dataset sizes (adjust to your compute):
- m=1: 10k to 50k
- m=2: 20k to 100k
- m=3: 20k to 100k
- m=4: 10k to 50k
- m=5: 5k to 20k Split:
- Train 70%, validation 15%, test 15% (stratify if you bin into good/bad). ### 6.2 Targets and metrics (pick one, or do both)
- Regression: predict fitness value - Metrics: MSE, MAE, Spearman rank correlation (ranking matters)
- Classification: predict “good” if in top X% (example: top 10%) - Metrics: accuracy, F1, precision@k (finding top candidates) ### 6.3 Features (start simple)
Two options:
1. **Raw bits** as features (works with linear models and trees, but large for m=5).
2. **Derived behavioral features** (more interpretable): - fraction of states that cooperate - “forgiveness”: cooperate after CD or DD patterns - “retaliation”: defect after DC patterns - symmetry or bias (how often you punish vs forgive) - performance vs a small probe set (tiny set of opponents) as cheap features ### 6.4 ML models to compare (keep it modest)
Compare 2 to 4 models:
- Logistic regression (classification) or ridge regression (regression)
- Random forest
- Gradient boosting (XGBoost / LightGBM if allowed, otherwise sklearn GradientBoosting)
- Small MLP Report:
- which model predicts rankings best
- how prediction quality changes with memory depth ### 6.5 How ML ties back to optimization (optional)
Two-stage evaluation:
- ML predicts top candidates
- only fully evaluate the predicted top K% with real tournaments This can save compute, but you must report any bias introduced. --- ## Step 7: Pattern extraction (turn bitstrings into “rules”)
Your report should not just dump a bitstring. Extract behavior. For the best strategies per memory depth:
- Compute “cooperation map” over states and summarize: - when does it cooperate after CC streaks? - how does it react to single defection vs repeated defection? - does it recover from mutual defection (DD loops)?
- Test each best strategy head-to-head vs TFT, TF2T, STFT: - show payoff and qualitative behavior (does it settle into CC, alternate, or fall into DD?) For memory-1, you can summarize as a simple 4-rule table:
- action after CC
- action after CD
- action after DC
- action after DD For deeper memory, summarize by grouping states:
- “after any DD in last m rounds”
- “after exactly one opponent defection in last k rounds”
- “after mutual cooperation streak length >= s” --- ## Step 8: Report checklist (what you must be able to say clearly)
- What payoff matrix, horizon, noise model did you use?
- What opponent pool defines “good” for fitness?
- For each optimizer: what parameters did you sweep and why?
- For each memory depth m: what was best, and how stable was it across seeds?
- Comparison against TFT, TF2T, STFT (and your other baselines).
- ML section (if included): dataset construction, models compared, metrics, and results.
- Patterns: what do strong strategies share? --- ## Quick acceptance criteria (to know you are done)
- You can run one command/script and produce: - results for GA, EDA, HC-RR, Tabu - for m = 1, 2, 3, 4, 5 - with saved best strategies and summary plots/tables
- You can reproduce the best result for a setting using the same random seed.
- You have head-to-head comparisons vs TFT, TF2T, STFT.
- You can describe at least 2 patterns found in strong strategies.