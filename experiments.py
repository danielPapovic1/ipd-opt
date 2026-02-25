"""
Experiments and Parameter Tuning for IPD Optimization
=====================================================
Runs comprehensive experiments comparing different methods and parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import random
from typing import List, Dict, Tuple
from dataclasses import asdict
import pandas as pd
import os
from ipd_core import (
    IPDGame, Strategy, create_strategy_from_bitstring,
    generate_random_strategy, REFERENCE_STRATEGIES, 
    TFT, TF2T, STFT, ALLD, ALLC, RAND, GRIM, PAVLOV
)
from optimization import (
    GeneticAlgorithm, EDA, HillClimbing, TabuSearch,
    FitnessEvaluator, OptimizationResult
)
from ml_prediction import (
    generate_training_data, StrategyPredictor, 
    extract_features, analyze_patterns
)


class ExperimentRunner:
    """Runs comprehensive experiments on IPD optimization"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self.results_cache = {}
    
    def run_parameter_tuning_ga(self, 
                                 opponents: List[Strategy],
                                 generations: int = 500,
                                 use_tournament_fitness: bool = True,
                                 variance_penalty: float = 0.5,
                                 coevolution: bool = True,
                                 coevolution_k: int = 5) -> pd.DataFrame:
        """
        Test different GA parameters.
        Vary: population_size, mutation_rate
        """
        print("Running GA Parameter Tuning...")
        
        population_sizes = [50, 100, 200]
        mutation_rates = [0.001, 0.01, 0.05]
        
        results = []
        
        for pop_size in population_sizes:
            for mut_rate in mutation_rates:
                print(f"  Testing pop_size={pop_size}, mut_rate={mut_rate}")
                
                ga = GeneticAlgorithm(
                    population_size=pop_size,
                    mutation_rate=mut_rate,
                    memory_depth=1,
                    num_rounds=100,
                    use_tournament_fitness=use_tournament_fitness,
                    variance_penalty=variance_penalty,
                    coevolution=coevolution,
                    coevolution_k=coevolution_k
                )
                
                result = ga.evolve(opponents, generations=generations, verbose=False)
                
                results.append({
                    'method': 'GA',
                    'population_size': pop_size,
                    'mutation_rate': mut_rate,
                    'best_fitness': result.best_fitness,
                    'time_taken': result.time_taken,
                    'best_strategy': result.best_strategy.bitstring
                })
        
        return pd.DataFrame(results)
    
    def run_memory_depth_experiment(self,
                                     opponents: List[Strategy],
                                     generations: int = 500,
                                     use_tournament_fitness: bool = True,
                                     variance_penalty: float = 0.5) -> pd.DataFrame:
        """
        Test different memory depths for all optimization methods.
        """
        print("Running Memory Depth Experiment...")

        # Include required depths 3, 4, 5 (and keep 1, 2 for baseline)
        memory_depths = [1, 2, 3, 4, 5]
        results = []
        
        for depth in memory_depths:
            print(f"  Testing memory_depth={depth}")
            
            # GA
            ga = GeneticAlgorithm(population_size=100, mutation_rate=0.01, 
                                 memory_depth=depth,
                                 use_tournament_fitness=use_tournament_fitness,
                                 variance_penalty=variance_penalty,
                                 coevolution=True,
                                 coevolution_k=5)
            result_ga = ga.evolve(opponents, generations=generations, verbose=False)
            results.append({
                'method': 'GA',
                'memory_depth': depth,
                'best_fitness': result_ga.best_fitness,
                'time_taken': result_ga.time_taken,
                'strategy_bits': len(result_ga.best_strategy.bitstring)
            })
            
            # EDA
            eda = EDA(population_size=100, memory_depth=depth,
                      use_tournament_fitness=use_tournament_fitness,
                      variance_penalty=variance_penalty,
                      coevolution=True,
                      coevolution_k=5)
            result_eda = eda.evolve(opponents, generations=generations, verbose=False)
            results.append({
                'method': 'EDA',
                'memory_depth': depth,
                'best_fitness': result_eda.best_fitness,
                'time_taken': result_eda.time_taken,
                'strategy_bits': len(result_eda.best_strategy.bitstring)
            })
            
            # Hill Climbing
            hc = HillClimbing(max_iterations=generations, restarts=10, 
                            memory_depth=depth,
                            use_tournament_fitness=use_tournament_fitness,
                            variance_penalty=variance_penalty)
            result_hc = hc.evolve(opponents, verbose=False)
            results.append({
                'method': 'HillClimbing',
                'memory_depth': depth,
                'best_fitness': result_hc.best_fitness,
                'time_taken': result_hc.time_taken,
                'strategy_bits': len(result_hc.best_strategy.bitstring)
            })
            
            # Tabu Search
            ts = TabuSearch(max_iterations=generations, tabu_size=10, 
                          memory_depth=depth,
                          use_tournament_fitness=use_tournament_fitness,
                          variance_penalty=variance_penalty)
            result_ts = ts.evolve(opponents, verbose=False)
            results.append({
                'method': 'TabuSearch',
                'memory_depth': depth,
                'best_fitness': result_ts.best_fitness,
                'time_taken': result_ts.time_taken,
                'strategy_bits': len(result_ts.best_strategy.bitstring)
            })
        
        return pd.DataFrame(results)
    
    def compare_all_methods(self,
                           opponents: List[Strategy],
                           generations: int = 1000,
                           n_runs: int = 5,
                           base_seed: int = 42,
                           rounds_range: Tuple[int, int] = (80, 120),
                           use_tournament_fitness: bool = True,
                           variance_penalty: float = 0.5,
                           coevolution: bool = True,
                           coevolution_k: int = 5) -> pd.DataFrame:
        """
        Compare all optimization methods with multiple runs.
        """
        print(f"Comparing All Methods ({n_runs} runs each)...")
        
        results = []
        
        for run in range(n_runs):
            # Deterministic but different seed per run for variation
            run_seed = base_seed + run
            random.seed(run_seed)
            np.random.seed(run_seed)

            # Vary rounds per run to avoid identical fitness plateaus
            if rounds_range:
                num_rounds = random.randint(rounds_range[0], rounds_range[1])
            else:
                num_rounds = 100

            print(f"  Run {run + 1}/{n_runs}")
            
            # GA
            ga = GeneticAlgorithm(population_size=100, mutation_rate=0.01, num_rounds=num_rounds,
                                  use_tournament_fitness=use_tournament_fitness,
                                  variance_penalty=variance_penalty,
                                  coevolution=coevolution,
                                  coevolution_k=coevolution_k)
            result_ga = ga.evolve(opponents, generations=generations)
            results.append({
                'run': run,
                'method': 'GA',
                'best_fitness': result_ga.best_fitness,
                'time_taken': result_ga.time_taken,
                'final_gen': len(result_ga.fitness_history)
            })
            
            # EDA
            eda = EDA(population_size=100, num_rounds=num_rounds,
                      use_tournament_fitness=use_tournament_fitness,
                      variance_penalty=variance_penalty,
                      coevolution=coevolution,
                      coevolution_k=coevolution_k)
            result_eda = eda.evolve(opponents, generations=generations)
            results.append({
                'run': run,
                'method': 'EDA',
                'best_fitness': result_eda.best_fitness,
                'time_taken': result_eda.time_taken,
                'final_gen': len(result_eda.fitness_history)
            })
            
            # Hill Climbing
            hc = HillClimbing(max_iterations=generations, restarts=10, num_rounds=num_rounds,
                              use_tournament_fitness=use_tournament_fitness,
                              variance_penalty=variance_penalty)
            result_hc = hc.evolve(opponents)
            results.append({
                'run': run,
                'method': 'HillClimbing',
                'best_fitness': result_hc.best_fitness,
                'time_taken': result_hc.time_taken,
                'final_gen': len(result_hc.fitness_history)
            })
            
            # Tabu Search
            ts = TabuSearch(max_iterations=generations, tabu_size=10, num_rounds=num_rounds,
                            use_tournament_fitness=use_tournament_fitness,
                            variance_penalty=variance_penalty)
            result_ts = ts.evolve(opponents)
            results.append({
                'run': run,
                'method': 'TabuSearch',
                'best_fitness': result_ts.best_fitness,
                'time_taken': result_ts.time_taken,
                'final_gen': len(result_ts.fitness_history)
            })
        
        return pd.DataFrame(results)
    
    def evolved_vs_reference(self,
                            opponents: List[Strategy],
                            generations: int = 1000,
                            use_tournament_fitness: bool = True,
                            variance_penalty: float = 0.5,
                            coevolution: bool = True,
                            coevolution_k: int = 5) -> pd.DataFrame:
        """
        Compare evolved strategies against reference strategies.
        """
        print("Comparing Evolved vs Reference Strategies...")
        
        # Evolve strategies with each method
        methods = {
            'GA': GeneticAlgorithm(population_size=100, mutation_rate=0.01,
                                   use_tournament_fitness=use_tournament_fitness,
                                   variance_penalty=variance_penalty,
                                   coevolution=coevolution,
                                   coevolution_k=coevolution_k),
            'EDA': EDA(population_size=100,
                       use_tournament_fitness=use_tournament_fitness,
                       variance_penalty=variance_penalty,
                       coevolution=coevolution,
                       coevolution_k=coevolution_k),
            'HillClimbing': HillClimbing(max_iterations=generations, restarts=10,
                                         use_tournament_fitness=use_tournament_fitness,
                                         variance_penalty=variance_penalty),
            'TabuSearch': TabuSearch(max_iterations=generations, tabu_size=10,
                                     use_tournament_fitness=use_tournament_fitness,
                                     variance_penalty=variance_penalty)
        }
        
        evolved = {}
        for name, optimizer in methods.items():
            print(f"  Evolving with {name}...")
            result = optimizer.evolve(opponents, generations=generations)
            evolved[name] = result.best_strategy
        
        # Run tournament with all strategies
        all_strategies = list(evolved.values()) + REFERENCE_STRATEGIES
        evolved_names = {s.name for s in evolved.values()}
        
        game = IPDGame()
        tournament_results = game.round_robin_tournament(all_strategies, 100)
        
        # Convert to DataFrame
        results = []
        for name, data in tournament_results.items():
            results.append({
                'strategy': name,
                'avg_score': data['avg_score'],
                'total_score': data['total_score'],
                'matches': data['matches'],
                'type': 'Evolved' if name in evolved_names else 'Reference'
            })
        
        return pd.DataFrame(results).sort_values('avg_score', ascending=False)
    
    def run_ml_experiments(self,
                          opponents: List[Strategy],
                          train_sizes: List[int] = [100, 500, 1000, 2000]) -> pd.DataFrame:
        """
        Test ML prediction with different training set sizes.
        """
        print("Running ML Experiments...")
        
        results = []
        
        for size in train_sizes:
            print(f"  Training with {size} samples...")
            
            X, y, feature_names, strategies, fitnesses = generate_training_data(
                n_samples=size,
                opponent_strategies=opponents,
                num_rounds=100,
                memory_depth=1
            )
            
            predictor = StrategyPredictor()
            ml_results = predictor.train(X, y, feature_names, test_size=0.2)
            
            for r in ml_results:
                results.append({
                    'train_size': size,
                    'model': r.model_name,
                    'accuracy': r.accuracy,
                    'precision': r.precision,
                    'recall': r.recall,
                    'f1': r.f1
                })
        
        return pd.DataFrame(results)


def create_visualizations(results_dict: Dict, output_dir: str = "."):
    """Create comprehensive visualizations"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Fitness convergence comparison
    if 'convergence' in results_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        for method, data in results_dict['convergence'].items():
            ax.plot(data['generations'], data['fitness'], label=method, linewidth=2)
        ax.set_xlabel('Generation/Iteration')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Convergence Comparison')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/convergence_comparison.png", dpi=150)
        plt.close()
    
    # 2. Method comparison boxplot
    if 'method_comparison' in results_dict:
        df = results_dict['method_comparison']
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column='best_fitness', by='method', ax=ax)
        ax.set_title('Best Fitness by Optimization Method')
        ax.set_xlabel('Method')
        ax.set_ylabel('Best Fitness')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/method_comparison.png", dpi=150)
        plt.close()

    # 2b. GA parameter heatmap
    if 'ga_params' in results_dict:
        df = results_dict['ga_params']
        pivot = df.pivot(index='population_size',
                         columns='mutation_rate',
                         values='best_fitness')
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(x) for x in pivot.index])
        ax.set_xlabel('Mutation Rate')
        ax.set_ylabel('Population Size')
        ax.set_title('GA Parameter Heatmap (Best Fitness)')
        fig.colorbar(im, ax=ax, label='Best Fitness')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ga_parameter_heatmap.png", dpi=150)
        plt.close()
    
    # 3. Memory depth impact
    if 'memory_depth' in results_dict:
        df = results_dict['memory_depth']
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in df['method'].unique():
            data = df[df['method'] == method]
            ax.plot(data['memory_depth'], data['best_fitness'], 
                   marker='o', label=method, linewidth=2)
        ax.set_xlabel('Memory Depth')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Impact of Memory Depth on Performance')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/memory_depth_impact.png", dpi=150)
        plt.close()
    
    # 4. ML model comparison
    if 'ml_results' in results_dict:
        df = results_dict['ml_results']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            for model in df['model'].unique():
                data = df[df['model'] == model]
                ax.plot(data['train_size'], data[metric], 
                       marker='o', label=model, linewidth=2)
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs Training Size')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ml_comparison.png", dpi=150)
        plt.close()

    # 5. Tournament results bar chart
    if 'tournament_results' in results_dict:
        df = results_dict['tournament_results'].copy()
        df = df.sort_values('avg_score', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = df['type'].map({'Evolved': '#2ca02c', 'Reference': '#1f77b4'}).fillna('#7f7f7f')
        ax.barh(df['strategy'], df['avg_score'], color=colors)
        ax.set_xlabel('Average Score')
        ax.set_title('Tournament Results (Evolved vs Reference)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tournament_results.png", dpi=150)
        plt.close()

    # 6. Evolved vs Reference comparison (aggregate)
    if 'tournament_results' in results_dict:
        df = results_dict['tournament_results'].copy()
        summary = df.groupby('type')['avg_score'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(summary['type'], summary['avg_score'], color=['#2ca02c', '#1f77b4'])
        ax.set_ylabel('Average Score')
        ax.set_title('Evolved vs Reference (Avg Score)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/evolved_comparison.png", dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def run_full_experiment_suite(output_dir: str = "results"):
    """Run the complete experiment suite"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("IPD OPTIMIZATION - FULL EXPERIMENT SUITE")
    print("=" * 70)
    
    runner = ExperimentRunner(output_dir)
    
    # Define opponents for evolution
    # Expanded opponent set to reduce overfitting
    opponents = [TFT, ALLD, ALLC, TF2T, STFT, GRIM, PAVLOV]
    
    all_results = {}
    
    # 1. Parameter tuning for GA
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: GA Parameter Tuning")
    print("=" * 70)
    ga_params = runner.run_parameter_tuning_ga(opponents, generations=300)
    print(ga_params.to_string())
    ga_params.to_csv(f"{output_dir}/ga_parameter_tuning.csv", index=False)
    all_results['ga_params'] = ga_params
    
    # 2. Memory depth experiment
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Memory Depth Comparison")
    print("=" * 70)
    memory_results = runner.run_memory_depth_experiment(opponents, generations=300)
    print(memory_results.to_string())
    memory_results.to_csv(f"{output_dir}/memory_depth_results.csv", index=False)
    all_results['memory_depth'] = memory_results
    
    # 3. Method comparison
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Method Comparison (5 runs)")
    print("=" * 70)
    method_comparison = runner.compare_all_methods(opponents, generations=500, n_runs=5)
    print(method_comparison.groupby('method').agg({
        'best_fitness': ['mean', 'std', 'min', 'max'],
        'time_taken': ['mean', 'std']
    }).to_string())
    method_comparison.to_csv(f"{output_dir}/method_comparison.csv", index=False)
    all_results['method_comparison'] = method_comparison
    
    # 4. Evolved vs Reference
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Evolved vs Reference Strategies")
    print("=" * 70)
    tournament_results = runner.evolved_vs_reference(opponents, generations=500)
    print(tournament_results.to_string())
    tournament_results.to_csv(f"{output_dir}/tournament_results.csv", index=False)
    all_results['tournament_results'] = tournament_results
    
    # 5. ML experiments
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: ML Prediction Experiments")
    print("=" * 70)
    ml_results = runner.run_ml_experiments(opponents, train_sizes=[200, 500, 1000])
    print(ml_results.to_string())
    ml_results.to_csv(f"{output_dir}/ml_results.csv", index=False)
    all_results['ml_results'] = ml_results
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating Visualizations...")
    print("=" * 70)
    create_visualizations(all_results, output_dir)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")
    
    return all_results


if __name__ == "__main__":
    results = run_full_experiment_suite()
