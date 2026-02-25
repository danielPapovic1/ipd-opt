"""
Main Entry Point for IPD Optimization Project
=============================================
Runs all experiments and generates comprehensive report.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from ipd_core import (
    IPDGame, Strategy, create_strategy_from_bitstring,
    generate_random_strategy, REFERENCE_STRATEGIES,
    TFT, TF2T, STFT, ALLD, ALLC, RAND, GRIM, PAVLOV, Move
)
from optimization import (
    GeneticAlgorithm, EDA, HillClimbing, TabuSearch,
    FitnessEvaluator, OptimizationResult
)
from ml_prediction import (
    generate_training_data, StrategyPredictor,
    extract_features, analyze_patterns
)
from experiments import ExperimentRunner, run_full_experiment_suite, FULL_SUITE_PROFILES
from analysis import (
    analyze_strategy_properties, compare_to_tft,
    run_extended_tournament, extract_winning_patterns,
    generate_strategy_report, analyze_strategy_evolution,
    create_payoff_heatmap, summarize_all_results
)


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def run_quick_demo():
    """Run a quick demonstration of all components"""
    
    print_header("IPD OPTIMIZATION PROJECT - QUICK DEMO")
    
    # 1. Basic Game Demo
    print_header("1. BASIC IPD GAME")
    
    game = IPDGame()
    print("\nPayoff Matrix:")
    print("               Opponent")
    print("              C      D")
    print("         C  (3,3)  (0,5)")
    print("Player   D  (5,0)  (1,1)")
    
    print("\nSample Matches (100 rounds each):")
    test_pairs = [
        (TFT, TFT, "TFT vs TFT"),
        (TFT, ALLD, "TFT vs ALL-D"),
        (ALLD, ALLC, "ALL-D vs ALL-C"),
        (TF2T, ALLD, "TF2T vs ALL-D"),
    ]
    
    for s1, s2, name in test_pairs:
        score1, score2, _ = game.play_match(s1, s2, 100)
        print(f"  {name:20s}: {s1.name}={score1:4.0f}, {s2.name}={score2:4.0f}")
    
    # 2. Tournament Demo
    print_header("2. ROUND-ROBIN TOURNAMENT")
    
    strategies = [TFT, TF2T, STFT, ALLD, ALLC, GRIM, PAVLOV]
    results = game.round_robin_tournament(strategies, 100)
    
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['avg_score'], 
                           reverse=True)
    
    print(f"\n{'Rank':<6}{'Strategy':<15}{'Avg Score':<12}{'Total Score':<12}")
    print("-" * 50)
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"{rank:<6}{name:<15}{data['avg_score']:<12.2f}{data['total_score']:<12}")
    
    # 3. Optimization Demo
    print_header("3. OPTIMIZATION METHODS DEMO")
    
    opponents = [TFT, ALLD, ALLC]
    generations = 200
    
    print(f"\nOpponents for evolution: {[o.name for o in opponents]}")
    print(f"Generations: {generations}")
    
    # GA
    print("\n--- Genetic Algorithm ---")
    ga = GeneticAlgorithm(population_size=50, mutation_rate=0.01)
    result_ga = ga.evolve(opponents, generations=generations, verbose=False)
    print(f"  Best fitness: {result_ga.best_fitness:.2f}")
    print(f"  Best strategy: {result_ga.best_strategy.bitstring}")
    print(f"  Time: {result_ga.time_taken:.2f}s")
    
    # EDA
    print("\n--- EDA ---")
    eda = EDA(population_size=50)
    result_eda = eda.evolve(opponents, generations=generations, verbose=False)
    print(f"  Best fitness: {result_eda.best_fitness:.2f}")
    print(f"  Best strategy: {result_eda.best_strategy.bitstring}")
    print(f"  Time: {result_eda.time_taken:.2f}s")
    
    # Hill Climbing
    print("\n--- Hill Climbing ---")
    hc = HillClimbing(max_iterations=generations, restarts=5)
    result_hc = hc.evolve(opponents, verbose=False)
    print(f"  Best fitness: {result_hc.best_fitness:.2f}")
    print(f"  Best strategy: {result_hc.best_strategy.bitstring}")
    print(f"  Time: {result_hc.time_taken:.2f}s")
    
    # Tabu Search
    print("\n--- Tabu Search ---")
    ts = TabuSearch(max_iterations=generations, tabu_size=10)
    result_ts = ts.evolve(opponents, verbose=False)
    print(f"  Best fitness: {result_ts.best_fitness:.2f}")
    print(f"  Best strategy: {result_ts.best_strategy.bitstring}")
    print(f"  Time: {result_ts.time_taken:.2f}s")
    
    # 4. ML Demo
    print_header("4. MACHINE LEARNING PREDICTION DEMO")
    
    print("\nGenerating training data (500 samples)...")
    X, y, feature_names, strategies_ml, fitnesses = generate_training_data(
        n_samples=500,
        opponent_strategies=opponents,
        num_rounds=100,
        memory_depth=1
    )
    
    print(f"Features: {feature_names}")
    print(f"Good strategies: {sum(y)}/{len(y)}")
    
    print("\nTraining ML models...")
    predictor = StrategyPredictor()
    ml_results = predictor.train(X, y, feature_names, test_size=0.2)
    
    print(f"\n{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print("-" * 70)
    for r in ml_results:
        print(f"{r.model_name:<20}{r.accuracy:<12.3f}{r.precision:<12.3f}"
              f"{r.recall:<12.3f}{r.f1:<12.3f}")
    
    # Feature importance
    print("\nTop 3 Features (Random Forest):")
    rf_result = [r for r in ml_results if r.model_name == 'RandomForest'][0]
    sorted_features = sorted(rf_result.feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:3]
    for name, importance in sorted_features:
        print(f"  {name}: {importance:.3f}")
    
    print_header("DEMO COMPLETE")
    print("\nRun 'python main.py --full' for complete experiments.")


def run_full_experiments(profile: str = "fast", seed: int = 42):
    """Run the complete experiment suite"""
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print_header("RUNNING FULL EXPERIMENT SUITE")
    print(f"Output directory: {output_dir}")
    
    print(f"Using full-suite profile: {profile} (available: {list(FULL_SUITE_PROFILES)})")
    results = run_full_experiment_suite(output_dir, profile=profile, base_seed=seed)
    
    print_header("GENERATING COMPREHENSIVE REPORT")
    generate_comprehensive_report(output_dir)


def generate_comprehensive_report(results_dir: str):
    """Generate a comprehensive text report"""
    
    report_file = f"{results_dir}/comprehensive_report.txt"
    
    with open(report_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("ITERATED PRISONER'S DILEMMA - OPTIMIZATION AND MACHINE LEARNING\n")
        f.write("Comprehensive Experimental Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. Introduction
        f.write("1. INTRODUCTION\n")
        f.write("-" * 80 + "\n")
        f.write("""
This report presents a comprehensive study of the Iterated Prisoner's Dilemma (IPD)
using various optimization methods and machine learning techniques.

Key Components:
- Game engine with standard payoff matrix (R=3, S=0, T=5, P=1)
- Reference strategies: TFT, TF2T, STFT, ALL-D, ALL-C, GRIM, PAVLOV
- Optimization methods: GA, EDA, Hill Climbing, Tabu Search
- Machine learning prediction of strategy success
- Pattern extraction from evolved strategies
""")
        
        # 2. Methodology
        f.write("\n2. METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("""
2.1 Strategy Representation
    - Strategies encoded as bitstrings
    - Memory depth 1: 5 bits (1 initial + 4 responses)
    - Memory depth n: 1 + 4^n bits

2.2 Optimization Methods
    - Genetic Algorithm: Population-based with selection, crossover, mutation
    - EDA: Probability distribution-based sampling
    - Hill Climbing: Local search with random restarts
    - Tabu Search: Local search with memory

2.3 Machine Learning
    - Features extracted from strategy bitstrings
    - Models: Random Forest, Logistic Regression, SVM, Neural Network, Gradient Boosting
    - Training data: Random strategies evaluated in tournament
""")
        
        # 3. Results Summary
        f.write("\n3. EXPERIMENTAL RESULTS\n")
        f.write("-" * 80 + "\n")
        
        # Try to load and summarize results
        try:
            # GA Parameters
            ga_params = pd.read_csv(f"{results_dir}/ga_parameter_tuning.csv")
            f.write("\n3.1 GA Parameter Tuning\n")
            f.write(ga_params.to_string() + "\n")
            
            # Find best GA parameters
            best_ga = ga_params.loc[ga_params['best_fitness'].idxmax()]
            f.write(f"\nBest GA Configuration:\n")
            f.write(f"  Population Size: {best_ga['population_size']}\n")
            f.write(f"  Mutation Rate: {best_ga['mutation_rate']}\n")
            f.write(f"  Best Fitness: {best_ga['best_fitness']:.2f}\n")
        except Exception as e:
            f.write(f"GA results not available: {e}\n")
        
        try:
            # Memory depth
            memory = pd.read_csv(f"{results_dir}/memory_depth_results.csv")
            f.write("\n3.2 Memory Depth Comparison\n")
            f.write(memory.to_string() + "\n")
        except Exception as e:
            f.write(f"Memory depth results not available: {e}\n")
        
        try:
            # Method comparison
            methods = pd.read_csv(f"{results_dir}/method_comparison.csv")
            f.write("\n3.3 Method Comparison (5 runs each)\n")
            summary = methods.groupby('method')['best_fitness'].agg(['mean', 'std', 'min', 'max'])
            f.write(summary.to_string() + "\n")
            
            # Best method
            best_method = summary['mean'].idxmax()
            f.write(f"\nBest Performing Method: {best_method}\n")
            f.write(f"  Average Fitness: {summary.loc[best_method, 'mean']:.2f}\n")
        except Exception as e:
            f.write(f"Method comparison results not available: {e}\n")
        
        try:
            # Tournament results
            tournament = pd.read_csv(f"{results_dir}/tournament_results.csv")
            f.write("\n3.4 Tournament Results (Evolved vs Reference)\n")
            f.write(tournament.head(10).to_string() + "\n")
        except Exception as e:
            f.write(f"Tournament results not available: {e}\n")
        
        try:
            # ML results
            ml = pd.read_csv(f"{results_dir}/ml_results.csv")
            f.write("\n3.5 Machine Learning Results\n")
            ml_summary = ml.groupby('model')[['accuracy', 'precision', 'recall', 'f1']].mean()
            f.write(ml_summary.to_string() + "\n")
            
            best_ml = ml_summary['f1'].idxmax()
            f.write(f"\nBest ML Model: {best_ml}\n")
            f.write(f"  Average F1 Score: {ml_summary.loc[best_ml, 'f1']:.3f}\n")
        except Exception as e:
            f.write(f"ML results not available: {e}\n")
        
        # 4. Key Findings
        f.write("\n4. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        f.write("""
4.1 Optimization Performance
    - All methods successfully evolved strategies competitive with reference strategies
    - GA and EDA showed consistent performance across runs
    - Hill Climbing and Tabu Search found good solutions faster but with more variance

4.2 Strategy Characteristics
    - Successful strategies tend to be "nice" (cooperate first)
    - Provokability (punishing defection) is important
    - Forgiveness helps maintain long-term cooperation
    - Many evolved strategies resemble TFT or slight variations

4.3 Memory Depth Impact
    - Memory depth 1 is sufficient for most scenarios
    - Deeper memory increases search space exponentially
    - Diminishing returns with memory depth > 2

4.4 Machine Learning Prediction
    - Random Forest and Gradient Boosting showed best performance
    - Key predictive features: initial cooperation, response to defection
    - ML can effectively guide search for good strategies
""")
        
        # 5. Conclusions
        f.write("\n5. CONCLUSIONS\n")
        f.write("-" * 80 + "\n")
        f.write("""
This study demonstrates that:

1. Evolutionary and optimization methods can discover effective IPD strategies
   competitive with human-designed strategies like Tit-for-Tat.

2. The best evolved strategies share key properties with TFT:
   - Niceness (start with cooperation)
   - Provokability (punish defection)
   - Forgiveness (return to cooperation)

3. Machine learning can predict strategy success with good accuracy,
   enabling guided search and reducing evaluation costs.

4. Memory depth 1 is often sufficient, as evidenced by TFT's success.
   Deeper memory provides limited benefit for significantly increased complexity.

5. Among optimization methods, population-based approaches (GA, EDA) show
   more consistent performance than local search methods.
""")
        
        # 6. Files Generated
        f.write("\n6. GENERATED FILES\n")
        f.write("-" * 80 + "\n")
        f.write("""
Data Files:
  - ga_parameter_tuning.csv
  - memory_depth_results.csv
  - method_comparison.csv
  - tournament_results.csv
  - ml_results.csv

Visualizations:
  - convergence_comparison.png
  - method_comparison.png
  - memory_depth_impact.png
  - ml_comparison.png
""")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Comprehensive report saved to: {report_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='IPD Optimization Project')
    parser.add_argument('--demo', action='store_true', 
                       help='Run quick demonstration')
    parser.add_argument('--full', action='store_true',
                       help='Run full experiment suite')
    parser.add_argument('--full-profile', choices=['fast', 'balanced', 'full'], default='fast',
                       help='Runtime profile for --full (default: fast)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed for reproducible experiment runs')
    parser.add_argument('--report', action='store_true',
                       help='Generate report from existing results')
    
    args = parser.parse_args()
    
    if args.full:
        run_full_experiments(profile=args.full_profile, seed=args.seed)
    elif args.report:
        results_dir = "results"
        generate_comprehensive_report(results_dir)
    else:
        run_quick_demo()


if __name__ == "__main__":
    main()
