# Iterated Prisoner's Dilemma Optimization Project

## Project Overview

This repository studies strategy optimization in the Iterated Prisoner’s Dilemma (IPD) using four optimization methods (GA, EDA, Hill Climbing, Tabu Search), benchmark strategy tournaments, and ML-based strategy-success prediction.

The implementation aligns with the specification baseline in the provided blueprint:
- Standard payoff matrix is implemented as **R=3, S=0, T=5, P=1**.
- Iterated simulation supports configurable round counts (with defaults used at 100 and 200 in experiments/evaluation paths).
- Baseline strategy set includes TFT, TF2T, STFT, ALL-D, ALL-C, RAND (plus GRIM and PAVLOV).
- GA includes roulette-wheel selection, single-point crossover, and bit-flip mutation.
- Strategy encoding follows **1 + 4^n** bits for memory depth `n`.
- The experiment suite compares GA, EDA, Hill Climbing, and Tabu Search.
- ML integration trains 5 classifiers over extracted strategy features.

## Directory Map

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

## System Flow (Text Diagram)

1. **Inputs**: CLI flags in `main.py` (e.g., `--demo`, `--full`, `--report`) trigger the run mode.
2. **Simulation** (`ipd_core.py`): `IPDGame.play_match()` simulates repeated interactions using `PAYOFF_MATRIX`.
3. **Scoring** (`optimization.py`): `FitnessEvaluator.evaluate()` computes fitness and reuses cached match/fitness values.
4. **Optimization** (`optimization.py`): GA/EDA/HC/TS evolve candidate strategies and return `OptimizationResult`.
5. **Machine Learning** (`ml_prediction.py`): training data is generated from simulated strategy fitness, feature vectors are extracted, then multiple classifiers are trained.
6. **Analysis and Outputs** (`experiments.py`, `analysis.py`, `main.py`, `zd_analysis.py`): dataframes, plots, ANOVA/effect-size outputs, Pareto frontier outputs, ZD diagnostics, and report files are produced under `results/`.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

### 1) Run the Quick Demonstration
Provides a fast, low-generation run to prove the environment works without taking hours.

```bash
python main.py --demo
```

**Expected Output:** Prints a sample Payoff Matrix, quick match results, a small tournament ranking, short optimization run outputs (Best fitness, bits), and a fast ML training loop output to the console.

### 2) Run the Full Experiment Suite
Executes the comprehensive pipeline (GA tuning, memory depth tests, method comparisons, ML scale tests).

```bash
python main.py --full
```

**Expected Output:** Generates all CSV files, JSON metadata, and PNG plots in the results/ directory, concluding with the generation of comprehensive_report.txt.

### 3) Generate Report Only (From Existing Results)
If results are already in the results/ folder, this skips simulation and re-compiles the final text report.

```bash
python main.py --report
```

**Expected Output:** Rewrites results/comprehensive_report.txt based on the CSVs present.

## Engineering Strengths

- **Modular architecture:** game engine (`ipd_core.py`) is separated from search logic (`optimization.py`).
- **High-performance evaluation:** shared LRU-like caches for match-level and fitness-level reuse in `FitnessEvaluator`.
- **Statistical rigor:** ANOVA and Cohen’s d are included in `analysis.py`.
- **ZD diagnostics:** evolved strategies can be categorized as Non-ZD / Extortionate / Generous.
- **Reproducibility controls:** seeded run policies and metadata logging are included in `experiments.py` and `main.py`.
