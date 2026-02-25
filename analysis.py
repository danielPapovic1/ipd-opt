"""
Analysis and Pattern Extraction for IPD Strategies
==================================================
Analyzes evolved strategies and extracts patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter

from ipd_core import (
    IPDGame, Strategy, create_strategy_from_bitstring,
    Move, TFT, TF2T, STFT, ALLD, ALLC, RAND, GRIM, PAVLOV
)
from optimization import FitnessEvaluator


def analyze_strategy_properties(strategy: Strategy) -> Dict:
    """
    Analyze key properties of a strategy.
    
    Properties:
    - nice: cooperates on first move
    - provokable: punishes defection
    - forgiving: returns to cooperation after punishing
    - clear: easy to predict
    - robust: performs well against variety
    """
    props = {}
    
    if strategy.memory_depth == 1 and strategy.bitstring:
        bits = strategy.bitstring
        
        # Nice: cooperates first
        props['nice'] = bits[0] == '0'
        
        # Provokable: if opponent defects when I cooperate, do I defect?
        # Response to (C, D) - I cooperated, opponent defected
        props['provokable'] = bits[2] == '1'
        
        # Forgiving: after mutual defection, do I cooperate?
        props['forgiving_after_dd'] = bits[4] == '0'
        
        # Forgiving: after I defect and opponent cooperates, do I cooperate?
        props['forgiving_after_dc'] = bits[3] == '0'
        
        # TFT-like: matches TFT pattern
        props['is_tft'] = bits == '00101'
        
        # ALL-C like
        props['is_allc'] = bits == '00000'
        
        # ALL-D like
        props['is_alld'] = bits == '11111'
        
        # Cooperation rate
        props['coop_rate'] = bits.count('0') / len(bits)
        
        # Response to cooperation (when I cooperated)
        props['response_to_cc'] = 'C' if bits[1] == '0' else 'D'
        
        # Response to defection (when I cooperated)
        props['response_to_cd'] = 'C' if bits[2] == '0' else 'D'
    
    return props


def compare_to_tft(strategy: Strategy) -> Dict:
    """Compare a strategy to Tit-for-Tat"""
    tft_bits = "00101"  # TFT bitstring
    
    if not strategy.bitstring or len(strategy.bitstring) != 5:
        return {'similarity': 0.0}
    
    bits = strategy.bitstring
    
    # Hamming distance
    differences = sum(1 for a, b in zip(bits, tft_bits) if a != b)
    similarity = 1 - (differences / len(bits))
    
    return {
        'similarity': similarity,
        'differences': differences,
        'same_initial': bits[0] == tft_bits[0],
        'same_cc_response': bits[1] == tft_bits[1],
        'same_cd_response': bits[2] == tft_bits[2],
        'same_dc_response': bits[3] == tft_bits[3],
        'same_dd_response': bits[4] == tft_bits[4]
    }


def run_extended_tournament(strategies: List[Strategy], 
                            num_rounds: int = 100) -> pd.DataFrame:
    """
    Run extended tournament with detailed results.
    """
    game = IPDGame()
    n = len(strategies)
    
    # Initialize results matrix
    payoff_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            score_i, score_j, _ = game.play_match(strategies[i], strategies[j], num_rounds)
            payoff_matrix[i, j] = score_i
    
    # Create DataFrame
    strategy_names = [s.name for s in strategies]
    df = pd.DataFrame(payoff_matrix, index=strategy_names, columns=strategy_names)
    
    # Add summary statistics
    df['Average'] = df.mean(axis=1)
    df['Min'] = df.min(axis=1)
    df['Max'] = df.max(axis=1)
    df['Std'] = df.std(axis=1)
    
    return df


def extract_winning_patterns(strategies: List[Strategy], 
                             fitnesses: List[float],
                             top_percentile: float = 20) -> Dict:
    """
    Extract patterns from top-performing strategies.
    """
    threshold = np.percentile(fitnesses, 100 - top_percentile)
    
    top_strategies = [s for s, f in zip(strategies, fitnesses) if f >= threshold]
    bottom_strategies = [s for s, f in zip(strategies, fitnesses) if f < threshold]
    
    analysis = {
        'top_count': len(top_strategies),
        'bottom_count': len(bottom_strategies),
        'threshold': threshold,
        'patterns': {}
    }
    
    # Analyze properties
    top_props = [analyze_strategy_properties(s) for s in top_strategies if s.bitstring]
    bottom_props = [analyze_strategy_properties(s) for s in bottom_strategies if s.bitstring]
    
    if top_props and bottom_props:
        for key in top_props[0].keys():
            if isinstance(top_props[0][key], bool):
                top_pct = sum(1 for p in top_props if p[key]) / len(top_props) * 100
                bottom_pct = sum(1 for p in bottom_props if p[key]) / len(bottom_props) * 100
                analysis['patterns'][key] = {
                    'top_percentage': top_pct,
                    'bottom_percentage': bottom_pct,
                    'difference': top_pct - bottom_pct
                }
            else:
                top_avg = np.mean([p[key] for p in top_props])
                bottom_avg = np.mean([p[key] for p in bottom_props])
                analysis['patterns'][key] = {
                    'top_average': top_avg,
                    'bottom_average': bottom_avg,
                    'difference': top_avg - bottom_avg
                }
    
    # Most common bitstrings
    top_bitstrings = [s.bitstring for s in top_strategies if s.bitstring]
    bitstring_counts = Counter(top_bitstrings)
    analysis['common_strategies'] = bitstring_counts.most_common(10)
    
    return analysis


def generate_strategy_report(strategy: Strategy, 
                            opponents: List[Strategy]) -> str:
    """
    Generate a detailed report for a strategy.
    """
    game = IPDGame()
    evaluator = FitnessEvaluator(opponents, 100)
    
    report = []
    report.append("=" * 60)
    report.append(f"STRATEGY REPORT: {strategy.name}")
    report.append("=" * 60)
    
    # Basic info
    report.append(f"\nBitstring: {strategy.bitstring}")
    report.append(f"Memory Depth: {strategy.memory_depth}")
    
    # Properties
    props = analyze_strategy_properties(strategy)
    report.append("\n--- Properties ---")
    for key, value in props.items():
        report.append(f"  {key}: {value}")
    
    # Comparison to TFT
    tft_comparison = compare_to_tft(strategy)
    report.append("\n--- Comparison to TFT ---")
    for key, value in tft_comparison.items():
        report.append(f"  {key}: {value}")
    
    # Performance against opponents
    report.append("\n--- Performance Against Opponents ---")
    for opp in opponents:
        score, opp_score, _ = game.play_match(strategy, opp, 100)
        report.append(f"  vs {opp.name:10s}: {score:4.0f} - {opp_score:4.0f}")
    
    # Overall fitness
    fitness = evaluator.evaluate(strategy)
    report.append(f"\nOverall Fitness: {fitness:.2f}")
    
    return "\n".join(report)


def analyze_strategy_evolution(fitness_history: List[float],
                               generation_history: List[int]) -> Dict:
    """
    Analyze how fitness evolved over generations.
    """
    analysis = {
        'initial_fitness': fitness_history[0],
        'final_fitness': fitness_history[-1],
        'improvement': fitness_history[-1] - fitness_history[0],
        'improvement_pct': ((fitness_history[-1] - fitness_history[0]) / 
                           max(1, fitness_history[0]) * 100),
        'max_fitness': max(fitness_history),
        'min_fitness': min(fitness_history),
        'convergence_gen': None
    }
    
    # Find convergence generation (when fitness stops improving significantly)
    threshold = 0.01 * (max(fitness_history) - min(fitness_history))
    for i in range(1, len(fitness_history)):
        if abs(fitness_history[i] - fitness_history[i-1]) < threshold:
            analysis['convergence_gen'] = generation_history[i]
            break
    
    # Calculate rate of improvement
    if len(fitness_history) > 1:
        analysis['avg_improvement_per_gen'] = (fitness_history[-1] - fitness_history[0]) / len(fitness_history)
    
    return analysis


def create_payoff_heatmap(strategies: List[Strategy], 
                          output_file: str = None) -> pd.DataFrame:
    """
    Create a payoff heatmap for all strategy pairs.
    """
    df = run_extended_tournament(strategies)
    
    if output_file:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.iloc[:, :-4], annot=True, fmt='.0f', cmap='RdYlGn',
                   cbar_kws={'label': 'Payoff'})
        plt.title('Payoff Matrix - Round Robin Tournament')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
    
    return df


def summarize_all_results(results_dir: str = ".") -> str:
    """
    Create a comprehensive summary of all experimental results.
    """
    summary = []
    
    summary.append("=" * 70)
    summary.append("COMPREHENSIVE RESULTS SUMMARY")
    summary.append("=" * 70)
    
    # Load results
    try:
        ga_params = pd.read_csv(f"{results_dir}/ga_parameter_tuning.csv")
        summary.append("\n--- GA Parameter Tuning ---")
        summary.append(ga_params.to_string())
    except:
        pass
    
    try:
        memory = pd.read_csv(f"{results_dir}/memory_depth_results.csv")
        summary.append("\n--- Memory Depth Results ---")
        summary.append(memory.to_string())
    except:
        pass
    
    try:
        methods = pd.read_csv(f"{results_dir}/method_comparison.csv")
        summary.append("\n--- Method Comparison Summary ---")
        summary.append(methods.groupby('method')['best_fitness'].agg(['mean', 'std', 'min', 'max']).to_string())
    except:
        pass
    
    try:
        tournament = pd.read_csv(f"{results_dir}/tournament_results.csv")
        summary.append("\n--- Tournament Results (Top 10) ---")
        summary.append(tournament.head(10).to_string())
    except:
        pass
    
    try:
        ml = pd.read_csv(f"{results_dir}/ml_results.csv")
        summary.append("\n--- ML Results Summary ---")
        summary.append(ml.groupby('model')[['accuracy', 'f1']].mean().to_string())
    except:
        pass
    
    return "\n".join(summary)


if __name__ == "__main__":
    print("=" * 60)
    print("Strategy Analysis Test")
    print("=" * 60)
    
    # Test with reference strategies
    for s in [TFT, TF2T, ALLD, ALLC]:
        print(f"\n{s.name}:")
        props = analyze_strategy_properties(s)
        for k, v in props.items():
            print(f"  {k}: {v}")
    
    # Test TFT comparison
    print("\n" + "=" * 60)
    print("TFT Comparison Test")
    print("=" * 60)
    
    tft_clone = create_strategy_from_bitstring("00101", 1)
    comparison = compare_to_tft(tft_clone)
    print(f"TFT clone similarity: {comparison['similarity']}")
    
    # Test tournament
    print("\n" + "=" * 60)
    print("Extended Tournament")
    print("=" * 60)
    
    strategies = [TFT, TF2T, STFT, ALLD, ALLC, RAND, GRIM, PAVLOV]
    df = run_extended_tournament(strategies, 100)
    print(df.to_string())
