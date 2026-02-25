# Iterated Prisoner's Dilemma - Optimization Project

## Overview

This project implements a comprehensive study of the Iterated Prisoner's Dilemma (IPD) using various optimization methods and machine learning techniques to discover effective strategies.

## Project Structure

```
ipd_project/
├── ipd_core.py          # Core game engine and reference strategies
├── optimization.py      # Optimization algorithms (GA, EDA, HC, TS)
├── ml_prediction.py     # Machine learning for strategy prediction
├── experiments.py       # Experiment runner and parameter tuning
├── analysis.py          # Strategy analysis and pattern extraction
├── main.py              # Main entry point
├── results/             # Generated results and visualizations
└── README.md            # This file
```

## Components

### 1. IPD Core (`ipd_core.py`)
- Game engine with standard payoff matrix (R=3, S=0, T=5, P=1)
- Strategy representation as bitstrings
- Tournament system for evaluating strategies
- Reference strategies:
  - **TFT** (Tit-for-Tat): Cooperate first, then copy opponent
  - **TF2T** (Tit-for-Two-Tats): More forgiving variant
  - **STFT** (Suspicious TFT): Defect first, then copy
  - **ALL-D**: Always defect
  - **ALL-C**: Always cooperate
  - **GRIM**: Cooperate until defection, then always defect
  - **PAVLOV**: Win-stay, lose-shift

### 2. Optimization Methods (`optimization.py`)
- **Genetic Algorithm (GA)**: Population-based evolution
- **Estimation of Distribution Algorithm (EDA)**: Probability-based sampling
- **Hill Climbing**: Local search with random restarts
- **Tabu Search**: Local search with memory

### 3. Machine Learning (`ml_prediction.py`)
- Feature extraction from strategy bitstrings
- Models: Random Forest, Logistic Regression, SVM, Neural Network, Gradient Boosting
- Strategy success prediction
- Pattern analysis

### 4. Experiments (`experiments.py`)
- Parameter tuning for optimization methods
- Memory depth comparison
- Method comparison across multiple runs
- Evolved vs reference strategy tournaments
- ML training set size experiments

### 5. Analysis (`analysis.py`)
- Strategy property analysis (nice, provokable, forgiving)
- TFT comparison
- Extended tournament results
- Winning pattern extraction
- Strategy evolution analysis

## Usage

### Quick Demo
```bash
python main.py --demo
```

### Full Experiment Suite
```bash
python main.py --full
```

### Generate Report from Existing Results
```bash
python main.py --report
```

### Run Individual Components
```python
# Import modules
from ipd_core import *
from optimization import *
from ml_prediction import *

# Create game and run tournament
game = IPDGame()
strategies = [TFT, ALLD, ALLC, TF2T]
results = game.round_robin_tournament(strategies, num_rounds=100)

# Evolve a strategy using GA
ga = GeneticAlgorithm(population_size=100, mutation_rate=0.01)
result = ga.evolve([TFT, ALLD, ALLC], generations=1000)
print(f"Best strategy: {result.best_strategy.bitstring}")
print(f"Best fitness: {result.best_fitness}")

# Train ML predictor
X, y, feature_names, strategies, fitnesses = generate_training_data(
    n_samples=1000, opponent_strategies=[TFT, ALLD, ALLC]
)
predictor = StrategyPredictor()
ml_results = predictor.train(X, y, feature_names)
```

## Key Results

### Optimization Performance
- All methods successfully evolved strategies achieving fitness ~266 against [TFT, ALLD, ALLC]
- GA and EDA showed consistent performance
- Hill Climbing and Tabu Search found good solutions faster

### Tournament Results (Evolved vs Reference)
| Rank | Strategy | Avg Score | Type |
|------|----------|-----------|------|
| 1 | GRIM | 328.08 | Reference |
| 2 | ALL-D | 284.31 | Reference |
| 3 | TF2T | 274.00 | Reference |
| 4 | TFT | 273.54 | Reference |
| 5-8 | Evolved strategies | 256-257 | Evolved |

### Machine Learning Results
- Random Forest, SVM, Neural Network, Gradient Boosting: 100% accuracy
- Logistic Regression: 78-82% accuracy
- Key features: initial cooperation, response to defection

### Memory Depth Impact
- Memory depth 1: Best fitness = 266
- Memory depth 2: Best fitness = 235
- Conclusion: Deeper memory provides diminishing returns

## Implementation Details

### Strategy Encoding (Memory Depth 1)
```
Bit 0: Initial move (0=Cooperate, 1=Defect)
Bit 1: Response to (C,C)
Bit 2: Response to (C,D)
Bit 3: Response to (D,C)
Bit 4: Response to (D,D)

Example: "00101" = TFT
  - Initial: Cooperate (0)
  - After (C,C): Cooperate (0)
  - After (C,D): Defect (1)
  - After (D,C): Cooperate (0)
  - After (D,D): Defect (1)
```

### Payoff Matrix
```
               Opponent
              C      D
         C  (3,3)  (0,5)
Player   D  (5,0)  (1,1)
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Author

Generated for IPD Optimization Assignment
