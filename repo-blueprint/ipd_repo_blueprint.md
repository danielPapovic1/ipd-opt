## Section A: What Exists vs What Spec Claims

| Spec Requirement (from IPD-full-OVR.md) | Implemented? | Evidence (File Path + Anchor) | Notes |
|---|---|---|---|
| Prisoner's Dilemma Environment (R=3, S=0, T=5, P=1) | Yes | ipd_core.py -> PAYOFF_MATRIX | Standard Payoff Matrix is correctly hardcoded. |
| Iterated Game Simulation (100-200 rounds) | Yes | ipd_core.py -> IPDGame.play_match(num_rounds=100) | Supports configurable round limits, defaulting to 100 or 200 in evaluation. |
| Winner: Tit for Tat (TFT) | Yes | ipd_core.py -> tit_for_tat() / TFT | Acts as the primary baseline. |
| Variations: TF2T, STFT, ALLD, ALLC, RAND | Yes | ipd_core.py -> REFERENCE_STRATEGIES | Also adds GRIM and PAVLOV as extra baselines. |
| Genetic Algorithms (GA) (Pop=100, mutation=0.001, 1000 gens, crossover, roulette) | Yes | optimization.py -> GeneticAlgorithm.evolve() | Default mutation is 0.01 in code, but experiments test 0.001, 0.01, and 0.05. Uses single-point crossover and roulette wheel selection. |
| Strategy Encoding (Memory depth 1 = 5 bits, depth 3 = 65 bits) | Yes | ipd_core.py -> create_strategy_from_bitstring() | Formula implemented as 1 + 4^n bits. Correctly maps history combinations to action responses. |
| Experimentation: Test GA, EDA, Hill Climbing, Tabu Search | Yes | experiments.py -> compare_all_methods() | All four optimization classes are fully implemented and compared. |
| Memory Depth Experiments (depths 3, 4, 5) | Yes | experiments.py -> run_memory_depth_experiment() | Loops over depths 1 through 5 and records complexity vs. fitness. |
| Machine Learning Integration (Predict success, extract rules) | Yes | ml_prediction.py -> StrategyPredictor.train() | Extracts 9 features. Trains RF, LR, SVM, MLP, and GB models to predict fitness. |

---

## Section B: Repo System Map (Evidence-Backed)

### Directory Map (Plaintext)

```txt
ipd_project/
├── main.py                 # Primary CLI entry point
├── ipd_core.py             # Game engine, payoff matrix, baseline strategies
├── optimization.py         # GA, EDA, Hill Climbing, Tabu Search logic
├── ml_prediction.py        # ML feature extraction, training, and prediction
├── experiments.py          # Orchestration of comparative experiments
├── analysis.py             # Statistical tests, Pareto frontier, pattern extraction
├── zd_analysis.py          # Zero-Determinant strategy verification
├── requirements.txt        # Python dependencies
└── results/                # Output artifacts (CSVs, PNGs, JSONs, Reports)
```

### System Flow Diagram (Text Form)

- **Inputs:** User triggers experiments via CLI flags in main.py (e.g., --full).
- **Simulation (ipd_core.py):** IPDGame.play_match() generates a history of moves between two Strategy objects using the PAYOFF_MATRIX.
- **Scoring (optimization.py):** FitnessEvaluator.evaluate() calculates average scores over matches. Uses _GLOBAL_MATCH_CACHE to avoid re-simulating known bitstring pairings.
- **Optimization (optimization.py):** Search algorithms (like GeneticAlgorithm.evolve()) generate populations, fetch fitness scores from the Evaluator, and produce an OptimizationResult containing the best evolved Strategy.
- **Machine Learning (ml_prediction.py):** generate_training_data() creates random populations, scores them, and extracts features (extract_features()) to train 5 distinct classification models.
- **Analysis & Outputs (experiments.py & analysis.py):** ExperimentRunner aggregates the results, saves them to Pandas DataFrames (CSVs), plots matplotlib charts, and runs ANOVA stats / Pareto analysis before formatting comprehensive_report.txt.

---

## Section C: File-by-File Extraction Plan

### 1. ipd_core.py
- **Role:** The foundational rules engine and reference library.
- **Key Classes/Functions:** IPDGame, Strategy, PAYOFF_MATRIX, tit_for_tat(), create_strategy_from_bitstring().
- **What to say in docs:** Explain that this file isolates the standard game theory components. Detail how create_strategy_from_bitstring() translates a 64-bit string (for memory depth 3) into an action lookup table.
- **Connects to:** Used by every other file to evaluate interactions.

### 2. optimization.py
- **Role:** The evolutionary and local search engines.
- **Key Classes/Functions:** FitnessEvaluator, GeneticAlgorithm, EDA, HillClimbing, TabuSearch.
- **What to say in docs:** Describe the four distinct algorithms. Emphasize the FitnessEvaluator caching system (_GLOBAL_MATCH_CACHE), which is a major strength for speeding up generations. Detail the GA pipeline: Roulette Wheel Selection -> Single-Point Crossover -> Bit-flip Mutation.
- **Connects to:** Depends on ipd_core.py for playing the game. Called by experiments.py.

### 3. ml_prediction.py
- **Role:** Bridges game theory with supervised learning.
- **Key Classes/Functions:** extract_features(), StrategyPredictor.train(), StrategyPredictor.find_good_strategy().
- **What to say in docs:** Explain the 9 extracted features (e.g., initial_cooperate, response_to_cc, nice, provokable). Document that it uses Random Forest, Logistic Regression, SVM, Neural Network, and Gradient Boosting to classify if a strategy will fall into the top fitness percentile.
- **Connects to:** Depends on optimization.py to evaluate the training set.

### 4. experiments.py
- **Role:** The test orchestrator.
- **Key Classes/Functions:** ExperimentRunner.run_parameter_tuning_ga(), run_memory_depth_experiment(), compare_all_methods(), evolved_vs_reference(), run_ml_experiments().
- **What to say in docs:** Provide the logic behind the 5 major experiments. Document how it enforces reproducibility via _reset_random_state() and seeds.
- **Connects to:** Collects data from optimization.py and ml_prediction.py and writes to the results/ folder.

### 5. analysis.py & zd_analysis.py
- **Role:** Post-experiment statistical validation.
- **Key Classes/Functions:** statistical_significance_test() (ANOVA/Cohen's d), pareto_frontier_analysis(), is_zero_determinant().
- **What to say in docs:** Detail how the project doesn't just look at high scores, but runs strict statistical tests and categorizes Zero-Determinant (Extortionate vs Generous) behaviors.

---

## Section D: Results and Meaning Map

| Artifact Name | Generation Path | What it Means (IPD Context) | What it does NOT Mean |
|---|---|---|---|
| ga_parameter_tuning.csv & heatmap | experiments.py -> run_parameter_tuning_ga() | Shows how Population Size and Mutation Rate affect the max fitness achieved by the GA. | It does not dictate the absolute best parameters for all opponents, just the tested subset. |
| memory_depth_impact.png / pareto_frontier.csv | experiments.py -> run_memory_depth_experiment() | Maps Memory Depth (1-5) to Fitness. Shows diminishing returns (Pareto optimality) between strategy complexity and score. | It does not mean memory depth > 2 is useless, just less efficient for this specific evolutionary setup. |
| method_comparison.csv & boxplots | experiments.py -> compare_all_methods() | Compares the stability and peak performance of GA, EDA, Hill Climbing, and Tabu Search over multiple seeded runs. | It is not a definitive proof that EDA beats GA generally, only within these specific hyperparameters. |
| tournament_results.csv | experiments.py -> evolved_vs_reference() | Ranks the absolute best evolved strategy against human-designed baselines (TFT, GRIM, etc.) in a round-robin format. | High scores do not equal "invincibility"; they indicate adaptability to this specific pool of opponents. |
| ml_results.csv & ml_comparison.png | ml_prediction.py via run_ml_experiments() | Tracks Accuracy, Precision, Recall, and F1 score of ML models as training set size increases. Proves ML can predict strategy success. | 100% accuracy on depth=1 is expected due to limited search space (32 strategies), not magic AI. |

---

## Section E: Exact Reproduction Commands

### 1. Run the Quick Demonstration
Provides a fast, low-generation run to prove the environment works without taking hours.

```bash
python main.py --demo
```

**Expected Output:** Prints a sample Payoff Matrix, quick match results, a small tournament ranking, short optimization run outputs (Best fitness, bits), and a fast ML training loop output to the console.

### 2. Run the Full Experiment Suite
Executes the comprehensive pipeline (GA tuning, memory depth tests, method comparisons, ML scale tests).

```bash
python main.py --full
```

**Expected Output:** Generates all CSV files, JSON metadata, and PNG plots in the results/ directory, concluding with the generation of comprehensive_report.txt.

### 3. Generate Report Only (From Existing Results)
If results are already in the results/ folder, this skips simulation and re-compiles the final text report.

```bash
python main.py --report
```

**Expected Output:** Rewrites results/comprehensive_report.txt based on the CSVs present.

---

## Section F: Strengths Inventory (Repo-Grounded)

- **Clean, Modular Architecture:** The game engine (ipd_core.py) is completely decoupled from the search algorithms (optimization.py), allowing new algorithms to be added without touching the game logic.
- **High-Performance Evaluation:** Implements a thread-safe, memory-capped caching system (FitnessEvaluator._GLOBAL_MATCH_CACHE and _GLOBAL_FITNESS_CACHE in optimization.py) that prevents the engine from re-simulating known bitstring pairings.
- **Advanced Statistical Rigor:** Does not rely solely on raw scores. Includes ANOVA testing and Cohen's d effect size calculations (analysis.py -> statistical_significance_test()) to prove method superiority.
- **Zero-Determinant Analysis:** Automatically classifies evolved strategies as "Generous" or "Extortionate" ZD strategies (zd_analysis.py -> is_zero_determinant()) using least-squares fitting.
- **Strict Reproducibility:** The ExperimentRunner enforces deterministic seeds tied to the run index and algorithm type, and logs software environment metadata to reproducibility_metadata.json.

---

## Section G: “Now Generate the Docs” Instructions

Using this blueprint, you must generate a comprehensive 4-part documentation set for this repository.

1) Read Sections A and F to understand the project's goals, alignment to the original spec, and its engineering strengths.

2) Generate README.md (Project Overview):
- Use Section B to write the Directory Map and System Flow.
- Use Section E to write the "Quick Start" and reproduction commands exactly as formatted.

3) Generate ARCHITECTURE.md (System Design):
- Use Section C to outline the file-by-file component breakdown.
- Explain the Strategy bitstring encoding (1 + 4^n).
- Highlight the caching mechanism in optimization.py and the ML extraction features in ml_prediction.py.

4) Generate EXPERIMENTS.md (Results & Analysis):
- Use Section D to explain what every generated chart and CSV means.
- Explicitly mention the Pareto analysis and Zero-Determinant strategy classifications as outlined in Section F.

Requirement: You must trace every claim back to the evidence provided in this blueprint. Do not invent new features, algorithms, or CLI flags.
