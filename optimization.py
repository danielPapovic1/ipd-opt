"""
Optimization Algorithms for IPD Strategy Evolution
==================================================
Implements:
- Genetic Algorithm (GA)
- Estimation of Distribution Algorithm (EDA)
- Hill Climbing
- Tabu Search
"""

import random
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Any, ClassVar
from dataclasses import dataclass
from copy import deepcopy
import time

from ipd_core import (
    IPDGame, Strategy, create_strategy_from_bitstring,
    generate_random_strategy, REFERENCE_STRATEGIES, TFT, ALLD, ALLC
)


@dataclass
class OptimizationResult:
    """Results from an optimization run"""
    best_strategy: Strategy
    best_fitness: float
    fitness_history: List[float]
    generation_history: List[int]
    time_taken: float
    parameters: Dict
    method: str


class FitnessEvaluator:
    """Evaluates fitness of strategies in tournament or pairwise matches"""
    _MAX_MATCH_CACHE_SIZE: ClassVar[int] = 10000
    _MAX_FITNESS_CACHE_SIZE: ClassVar[int] = 10000
    _GLOBAL_MATCH_CACHE: ClassVar[OrderedDict] = OrderedDict()
    _GLOBAL_FITNESS_CACHE: ClassVar[OrderedDict] = OrderedDict()
    _GLOBAL_LOCK: ClassVar[threading.Lock] = threading.Lock()
    
    def __init__(self,
                 opponent_strategies: List[Strategy],
                 num_rounds: int = 100,
                 use_tournament_fitness: bool = False,
                 variance_penalty: float = 0.0,
                 parallel_eval: bool = False,
                 max_workers: Optional[int] = None):
        self.opponents = opponent_strategies
        self.num_rounds = num_rounds
        self.use_tournament_fitness = use_tournament_fitness
        self.variance_penalty = variance_penalty
        self.game = IPDGame()
        self.parallel_eval = parallel_eval
        self.max_workers = max_workers or os.cpu_count() or 1

    @classmethod
    def clear_global_caches(cls) -> None:
        """Clear shared evaluator caches to avoid unbounded growth across experiments."""
        with cls._GLOBAL_LOCK:
            cls._GLOBAL_MATCH_CACHE.clear()
            cls._GLOBAL_FITNESS_CACHE.clear()

    @classmethod
    def _cache_get(cls, cache: OrderedDict, key: Any) -> Optional[Any]:
        with cls._GLOBAL_LOCK:
            value = cache.get(key)
            if value is not None:
                cache.move_to_end(key)
            return value

    @classmethod
    def _cache_put(cls, cache: OrderedDict, key: Any, value: Any, max_size: int) -> None:
        with cls._GLOBAL_LOCK:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > max_size:
                cache.popitem(last=False)
    
    def _pairwise_scores(self, strategy: Strategy, opponents: List[Strategy]) -> List[float]:
        scores = []
        s_key = strategy.bitstring if strategy.is_bitstring and strategy.bitstring else strategy.name
        for opponent in opponents:
            o_key = opponent.bitstring if opponent.is_bitstring and opponent.bitstring else opponent.name
            key = (s_key, o_key, self.num_rounds)
            cached = FitnessEvaluator._cache_get(FitnessEvaluator._GLOBAL_MATCH_CACHE, key)
            if cached is not None:
                score = cached
            else:
                score, _, _ = self.game.play_match(strategy, opponent, self.num_rounds)
                FitnessEvaluator._cache_put(
                    FitnessEvaluator._GLOBAL_MATCH_CACHE,
                    key,
                    score,
                    FitnessEvaluator._MAX_MATCH_CACHE_SIZE
                )
            scores.append(score)
        return scores
    
    def evaluate(self, strategy: Strategy,
                 opponents: Optional[List[Strategy]] = None,
                 coevolution: bool = False) -> float:
        """
        Evaluate fitness by playing against opponents.
        Returns average score (optionally tournament-based) with variance penalty.
        """
        opponents = opponents if opponents is not None else self.opponents
        s_key = strategy.bitstring if strategy.is_bitstring and strategy.bitstring else strategy.name
        opp_keys = tuple(
            o.bitstring if o.is_bitstring and o.bitstring else o.name
            for o in opponents
        )
        f_key = (s_key, opp_keys, self.num_rounds, self.use_tournament_fitness, self.variance_penalty, coevolution)
        cached_fit = FitnessEvaluator._cache_get(FitnessEvaluator._GLOBAL_FITNESS_CACHE, f_key)
        if cached_fit is not None:
            return cached_fit

        scores = self._pairwise_scores(strategy, opponents)
        if not scores:
            return 0.0
        
        if self.use_tournament_fitness and not coevolution:
            all_strats = [strategy] + opponents
            results = self.game.round_robin_tournament(
                all_strats, self.num_rounds, include_self_play=False
            )
            base = results[strategy.name]['avg_score']
        else:
            base = sum(scores) / len(scores)
        
        if self.variance_penalty > 0:
            std = float(np.std(scores))
            fitness = base - self.variance_penalty * std
        else:
            fitness = base

        FitnessEvaluator._cache_put(
            FitnessEvaluator._GLOBAL_FITNESS_CACHE,
            f_key,
            fitness,
            FitnessEvaluator._MAX_FITNESS_CACHE_SIZE
        )
        return fitness
    
    def evaluate_population(self,
                            population: List[Strategy],
                            coevolution: bool = False,
                            coevolution_k: int = 5) -> List[float]:
        """Evaluate fitness for entire population"""
        if not coevolution:
            # ThreadPool overhead dominates very small populations.
            if not self.parallel_eval or len(population) < 10:
                return [self.evaluate(s) for s in population]
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                return list(ex.map(self.evaluate, population))
        
        fitnesses = []
        for s in population:
            # Dynamic opponents from population + fixed opponents
            pool = [p for p in population if p is not s]
            sampled = random.sample(pool, min(coevolution_k, len(pool))) if pool else []
            opponents = self.opponents + sampled
            fitnesses.append(self.evaluate(s, opponents=opponents, coevolution=True))
        return fitnesses


# ============== GENETIC ALGORITHM ==============

class GeneticAlgorithm:
    """
    Genetic Algorithm for evolving IPD strategies.
    
    Parameters:
    - population_size: Number of individuals in population
    - mutation_rate: Probability of bit flip mutation
    - crossover_rate: Probability of crossover
    - elitism: Number of best individuals to preserve
    - memory_depth: Memory depth for strategies
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8,
                 elitism: int = 2,
                 memory_depth: int = 1,
                 num_rounds: int = 100,
                 use_tournament_fitness: bool = False,
                 variance_penalty: float = 0.0,
                 coevolution: bool = False,
                 coevolution_k: int = 5):
        self.pop_size = population_size
        self.mut_rate = mutation_rate
        self.cross_rate = crossover_rate
        self.elitism = elitism
        self.memory_depth = memory_depth
        self.num_rounds = num_rounds
        self.use_tournament_fitness = use_tournament_fitness
        self.variance_penalty = variance_penalty
        self.coevolution = coevolution
        self.coevolution_k = coevolution_k
        
        # Calculate bitstring length
        if memory_depth == 1:
            self.bit_length = 5
        else:
            self.bit_length = 1 + (4 ** memory_depth)
        
        self.evaluator = None
    
    def initialize_population(self) -> List[Strategy]:
        """Create random initial population"""
        return [generate_random_strategy(self.memory_depth) 
                for _ in range(self.pop_size)]
    
    def roulette_wheel_selection(self, population: List[Strategy], 
                                  fitnesses: List[float]) -> Strategy:
        """Select individual using roulette wheel selection"""
        # Shift fitnesses to be non-negative
        min_fitness = min(fitnesses)
        adjusted = [f - min_fitness + 1e-10 for f in fitnesses]
        total_fitness = sum(adjusted)
        if total_fitness <= 0:
            return random.choice(population)

        pick = random.uniform(0, total_fitness)
        current = 0
        for individual, fitness in zip(population, adjusted):
            current += fitness
            if current >= pick:
                return individual
        return population[-1]
    
    def crossover(self, parent1: Strategy, parent2: Strategy) -> Tuple[Strategy, Strategy]:
        """Single-point crossover between two parents"""
        if random.random() > self.cross_rate:
            return parent1, parent2
        
        bits1, bits2 = parent1.bitstring, parent2.bitstring
        point = random.randint(1, len(bits1) - 1)
        
        child1_bits = bits1[:point] + bits2[point:]
        child2_bits = bits2[:point] + bits1[point:]
        
        child1 = create_strategy_from_bitstring(child1_bits, self.memory_depth)
        child2 = create_strategy_from_bitstring(child2_bits, self.memory_depth)
        
        return child1, child2
    
    def mutate(self, strategy: Strategy) -> Strategy:
        """Bit-flip mutation"""
        bits = list(strategy.bitstring)
        for i in range(len(bits)):
            if random.random() < self.mut_rate:
                bits[i] = '1' if bits[i] == '0' else '0'
        
        new_bits = ''.join(bits)
        return create_strategy_from_bitstring(new_bits, self.memory_depth)
    
    def evolve(self, 
               opponent_strategies: List[Strategy],
               generations: int = 1000,
               verbose: bool = False) -> OptimizationResult:
        """
        Run the genetic algorithm.
        
        Returns:
            OptimizationResult with best strategy and statistics
        """
        start_time = time.time()
        self.evaluator = FitnessEvaluator(
            opponent_strategies,
            self.num_rounds,
            use_tournament_fitness=self.use_tournament_fitness,
            variance_penalty=self.variance_penalty,
            parallel_eval=True
        )
        
        # Initialize population
        population = self.initialize_population()
        fitnesses = self.evaluator.evaluate_population(
            population,
            coevolution=self.coevolution,
            coevolution_k=self.coevolution_k
        )
        
        best_fitness_history = []
        generation_history = []
        
        best_ever = None
        best_fitness_ever = -float('inf')
        
        for gen in range(generations):
            # Track best
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness_ever:
                best_fitness_ever = fitnesses[max_idx]
                best_ever = deepcopy(population[max_idx])
            
            if verbose and gen % 100 == 0:
                print(f"Gen {gen:4d}: Best={best_fitness_ever:.2f}, "
                      f"Avg={np.mean(fitnesses):.2f}")
            
            best_fitness_history.append(best_fitness_ever)
            generation_history.append(gen)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(self.elitism):
                new_population.append(deepcopy(population[sorted_indices[i]]))
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = self.roulette_wheel_selection(population, fitnesses)
                parent2 = self.roulette_wheel_selection(population, fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            population = new_population
            fitnesses = self.evaluator.evaluate_population(
                population,
                coevolution=self.coevolution,
                coevolution_k=self.coevolution_k
            )
        
        time_taken = time.time() - start_time
        
        return OptimizationResult(
            best_strategy=best_ever,
            best_fitness=best_fitness_ever,
            fitness_history=best_fitness_history,
            generation_history=generation_history,
            time_taken=time_taken,
            parameters={
                'population_size': self.pop_size,
                'mutation_rate': self.mut_rate,
                'crossover_rate': self.cross_rate,
                'elitism': self.elitism,
                'memory_depth': self.memory_depth,
                'generations': generations
            },
            method='GA'
        )


# ============== ESTIMATION OF DISTRIBUTION ALGORITHM ==============

class EDA:
    """
    Estimation of Distribution Algorithm for IPD strategies.
    
    Uses a probability vector to sample new populations.
    """
    
    def __init__(self,
                 population_size: int = 100,
                 selection_rate: float = 0.3,
                 learning_rate: float = 0.1,
                 memory_depth: int = 1,
                 num_rounds: int = 100,
                 use_tournament_fitness: bool = False,
                 variance_penalty: float = 0.0,
                 coevolution: bool = False,
                 coevolution_k: int = 5):
        self.pop_size = population_size
        self.selection_rate = selection_rate
        self.learning_rate = learning_rate
        self.memory_depth = memory_depth
        self.num_rounds = num_rounds
        self.use_tournament_fitness = use_tournament_fitness
        self.variance_penalty = variance_penalty
        self.coevolution = coevolution
        self.coevolution_k = coevolution_k
        
        if memory_depth == 1:
            self.bit_length = 5
        else:
            self.bit_length = 1 + (4 ** memory_depth)
    
    def sample_population(self, prob_vector: np.ndarray) -> List[Strategy]:
        """Sample population from probability vector"""
        population = []
        for _ in range(self.pop_size):
            bits = ''.join('1' if random.random() < p else '0' 
                          for p in prob_vector)
            population.append(create_strategy_from_bitstring(bits, self.memory_depth))
        return population
    
    def update_prob_vector(self, prob_vector: np.ndarray, 
                          selected: List[Strategy]) -> np.ndarray:
        """Update probability vector based on selected individuals"""
        new_vector = np.zeros_like(prob_vector)
        
        for strategy in selected:
            for i, bit in enumerate(strategy.bitstring):
                new_vector[i] += 1 if bit == '1' else 0
        
        new_vector /= len(selected)
        
        # Smooth update
        return (1 - self.learning_rate) * prob_vector + self.learning_rate * new_vector
    
    def evolve(self,
               opponent_strategies: List[Strategy],
               generations: int = 1000,
               verbose: bool = False) -> OptimizationResult:
        """Run EDA optimization"""
        start_time = time.time()
        evaluator = FitnessEvaluator(
            opponent_strategies,
            self.num_rounds,
            use_tournament_fitness=self.use_tournament_fitness,
            variance_penalty=self.variance_penalty,
            parallel_eval=True
        )
        
        # Initialize probability vector (uniform)
        prob_vector = np.ones(self.bit_length) * 0.5
        
        population = self.sample_population(prob_vector)
        fitnesses = evaluator.evaluate_population(
            population,
            coevolution=self.coevolution,
            coevolution_k=self.coevolution_k
        )
        
        best_fitness_history = []
        generation_history = []
        
        best_ever = None
        best_fitness_ever = -float('inf')
        
        for gen in range(generations):
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness_ever:
                best_fitness_ever = fitnesses[max_idx]
                best_ever = deepcopy(population[max_idx])
            
            if verbose and gen % 100 == 0:
                print(f"Gen {gen:4d}: Best={best_fitness_ever:.2f}, "
                      f"Avg={np.mean(fitnesses):.2f}")
            
            best_fitness_history.append(best_fitness_ever)
            generation_history.append(gen)
            
            # Select top individuals
            num_select = int(self.pop_size * self.selection_rate)
            sorted_indices = np.argsort(fitnesses)[::-1][:num_select]
            selected = [population[i] for i in sorted_indices]
            
            # Update probability vector
            prob_vector = self.update_prob_vector(prob_vector, selected)
            
            # Sample new population
            population = self.sample_population(prob_vector)
            fitnesses = evaluator.evaluate_population(
                population,
                coevolution=self.coevolution,
                coevolution_k=self.coevolution_k
            )
        
        time_taken = time.time() - start_time
        
        return OptimizationResult(
            best_strategy=best_ever,
            best_fitness=best_fitness_ever,
            fitness_history=best_fitness_history,
            generation_history=generation_history,
            time_taken=time_taken,
            parameters={
                'population_size': self.pop_size,
                'selection_rate': self.selection_rate,
                'learning_rate': self.learning_rate,
                'memory_depth': self.memory_depth,
                'generations': generations
            },
            method='EDA'
        )


# ============== HILL CLIMBING ==============

class HillClimbing:
    """
    Hill Climbing with random restarts for IPD strategy optimization.
    """
    
    def __init__(self,
                 max_iterations: int = 1000,
                 restarts: int = 10,
                 memory_depth: int = 1,
                 num_rounds: int = 100,
                 use_tournament_fitness: bool = False,
                 variance_penalty: float = 0.0,
                 max_neighbors: Optional[int] = None):
        self.max_iter = max_iterations
        self.restarts = restarts
        self.memory_depth = memory_depth
        self.num_rounds = num_rounds
        self.use_tournament_fitness = use_tournament_fitness
        self.variance_penalty = variance_penalty
        # Sample neighbors by default for deep-memory strategies to avoid full neighborhood explosion.
        self.max_neighbors = 50 if max_neighbors is None and memory_depth > 2 else max_neighbors
        
        if memory_depth == 1:
            self.bit_length = 5
        else:
            self.bit_length = 1 + (4 ** memory_depth)
    
    def get_neighbors(self, strategy: Strategy) -> List[Strategy]:
        """Generate single-bit-flip neighbors (optionally sampled)"""
        neighbors = []
        bits = strategy.bitstring

        length = len(bits)
        if self.max_neighbors is not None and length > self.max_neighbors:
            positions = random.sample(range(length), self.max_neighbors)
        else:
            positions = range(length)

        for i in positions:
            new_bits = bits[:i] + ('1' if bits[i] == '0' else '0') + bits[i+1:]
            neighbors.append(create_strategy_from_bitstring(new_bits, self.memory_depth))
        
        return neighbors
    
    def evolve(self,
               opponent_strategies: List[Strategy],
               generations: Optional[int] = None,
               verbose: bool = False) -> OptimizationResult:
        """
        Run hill climbing with random restarts.

        The `generations` parameter is accepted for API compatibility with GA/EDA and
        is mapped to `max_iterations` when provided.
        """
        if generations is not None:
            self.max_iter = generations
        start_time = time.time()
        evaluator = FitnessEvaluator(
            opponent_strategies,
            self.num_rounds,
            use_tournament_fitness=self.use_tournament_fitness,
            variance_penalty=self.variance_penalty,
            parallel_eval=True
        )
        
        best_overall = None
        best_fitness_overall = -float('inf')
        
        fitness_history = []
        generation_history = []
        iteration = 0
        
        for restart in range(self.restarts):
            # Start from random point
            current = generate_random_strategy(self.memory_depth)
            current_fitness = evaluator.evaluate(current)
            
            improved = True
            local_iter = 0
            
            while improved and local_iter < self.max_iter // self.restarts:
                improved = False
                neighbors = self.get_neighbors(current)
                neighbor_fitnesses = evaluator.evaluate_population(neighbors)
                
                best_neighbor_idx = np.argmax(neighbor_fitnesses)
                best_neighbor_fitness = neighbor_fitnesses[best_neighbor_idx]
                
                if best_neighbor_fitness > current_fitness:
                    current = neighbors[best_neighbor_idx]
                    current_fitness = best_neighbor_fitness
                    improved = True
                
                if current_fitness > best_fitness_overall:
                    best_fitness_overall = current_fitness
                    best_overall = deepcopy(current)
                
                fitness_history.append(best_fitness_overall)
                generation_history.append(iteration)
                iteration += 1
                local_iter += 1
            
            if verbose:
                print(f"Restart {restart + 1}/{self.restarts}: "
                      f"Best={best_fitness_overall:.2f}")
        
        time_taken = time.time() - start_time
        
        return OptimizationResult(
            best_strategy=best_overall,
            best_fitness=best_fitness_overall,
            fitness_history=fitness_history,
            generation_history=generation_history,
            time_taken=time_taken,
            parameters={
                'max_iterations': self.max_iter,
                'restarts': self.restarts,
                'memory_depth': self.memory_depth
            },
            method='HillClimbing'
        )


# ============== TABU SEARCH ==============

class TabuSearch:
    """
    Tabu Search for IPD strategy optimization.
    """
    
    def __init__(self,
                 max_iterations: int = 1000,
                 tabu_size: int = 10,
                 memory_depth: int = 1,
                 num_rounds: int = 100,
                 use_tournament_fitness: bool = False,
                 variance_penalty: float = 0.0,
                 max_neighbors: Optional[int] = None):
        self.max_iter = max_iterations
        self.tabu_size = tabu_size
        self.memory_depth = memory_depth
        self.num_rounds = num_rounds
        self.use_tournament_fitness = use_tournament_fitness
        self.variance_penalty = variance_penalty
        # Sample neighbors by default for deep-memory strategies to avoid full neighborhood explosion.
        self.max_neighbors = 50 if max_neighbors is None and memory_depth > 2 else max_neighbors
        
        if memory_depth == 1:
            self.bit_length = 5
        else:
            self.bit_length = 1 + (4 ** memory_depth)
    
    def get_neighbors(self, strategy: Strategy) -> List[Strategy]:
        """Generate single-bit-flip neighbors (optionally sampled)"""
        neighbors = []
        bits = strategy.bitstring

        length = len(bits)
        if self.max_neighbors is not None and length > self.max_neighbors:
            positions = random.sample(range(length), self.max_neighbors)
        else:
            positions = range(length)

        for i in positions:
            new_bits = bits[:i] + ('1' if bits[i] == '0' else '0') + bits[i+1:]
            neighbors.append(create_strategy_from_bitstring(new_bits, self.memory_depth))
        
        return neighbors
    
    def evolve(self,
               opponent_strategies: List[Strategy],
               generations: Optional[int] = None,
               verbose: bool = False) -> OptimizationResult:
        """
        Run tabu search.

        The `generations` parameter is accepted for API compatibility with GA/EDA and
        is mapped to `max_iterations` when provided.
        """
        if generations is not None:
            self.max_iter = generations
        start_time = time.time()
        evaluator = FitnessEvaluator(
            opponent_strategies,
            self.num_rounds,
            use_tournament_fitness=self.use_tournament_fitness,
            variance_penalty=self.variance_penalty,
            parallel_eval=True
        )
        
        # Initialize
        current = generate_random_strategy(self.memory_depth)
        current_fitness = evaluator.evaluate(current)
        
        best_solution = deepcopy(current)
        best_fitness = current_fitness
        
        tabu_list = [current.bitstring]
        
        fitness_history = [best_fitness]
        generation_history = [0]
        
        for iteration in range(1, self.max_iter):
            neighbors = self.get_neighbors(current)
            neighbor_fitnesses = evaluator.evaluate_population(neighbors)
            
            # Find best non-tabu neighbor (or aspirational)
            best_neighbor = None
            best_neighbor_fitness = -float('inf')
            
            for neighbor, fitness in zip(neighbors, neighbor_fitnesses):
                is_tabu = neighbor.bitstring in tabu_list
                
                # Aspiration: accept if better than best ever
                if is_tabu and fitness <= best_fitness:
                    continue
                
                if fitness > best_neighbor_fitness:
                    best_neighbor_fitness = fitness
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                # All neighbors tabu, pick random
                random_idx = random.randrange(len(neighbors))
                best_neighbor = neighbors[random_idx]
                best_neighbor_fitness = neighbor_fitnesses[random_idx]
            
            # Move to best neighbor
            current = best_neighbor
            current_fitness = best_neighbor_fitness
            
            # Update tabu list
            tabu_list.append(current.bitstring)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)
            
            # Update best
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = deepcopy(current)
            
            fitness_history.append(best_fitness)
            generation_history.append(iteration)
            
            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:4d}: Best={best_fitness:.2f}")
        
        time_taken = time.time() - start_time
        
        return OptimizationResult(
            best_strategy=best_solution,
            best_fitness=best_fitness,
            fitness_history=fitness_history,
            generation_history=generation_history,
            time_taken=time_taken,
            parameters={
                'max_iterations': self.max_iter,
                'tabu_size': self.tabu_size,
                'memory_depth': self.memory_depth
            },
            method='TabuSearch'
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Optimization Algorithms Test")
    print("=" * 60)
    
    # Test opponents
    opponents = [TFT, ALLD, ALLC]
    
    print("\n1. Testing Genetic Algorithm:")
    print("-" * 50)
    ga = GeneticAlgorithm(population_size=50, mutation_rate=0.01, memory_depth=1)
    result_ga = ga.evolve(opponents, generations=200, verbose=True)
    print(f"Best fitness: {result_ga.best_fitness:.2f}")
    print(f"Best strategy: {result_ga.best_strategy.bitstring}")
    print(f"Time: {result_ga.time_taken:.2f}s")
    
    print("\n2. Testing EDA:")
    print("-" * 50)
    eda = EDA(population_size=50, memory_depth=1)
    result_eda = eda.evolve(opponents, generations=200, verbose=True)
    print(f"Best fitness: {result_eda.best_fitness:.2f}")
    print(f"Best strategy: {result_eda.best_strategy.bitstring}")
    print(f"Time: {result_eda.time_taken:.2f}s")
    
    print("\n3. Testing Hill Climbing:")
    print("-" * 50)
    hc = HillClimbing(max_iterations=500, restarts=5, memory_depth=1)
    result_hc = hc.evolve(opponents, verbose=True)
    print(f"Best fitness: {result_hc.best_fitness:.2f}")
    print(f"Best strategy: {result_hc.best_strategy.bitstring}")
    print(f"Time: {result_hc.time_taken:.2f}s")
    
    print("\n4. Testing Tabu Search:")
    print("-" * 50)
    ts = TabuSearch(max_iterations=500, tabu_size=10, memory_depth=1)
    result_ts = ts.evolve(opponents, verbose=True)
    print(f"Best fitness: {result_ts.best_fitness:.2f}")
    print(f"Best strategy: {result_ts.best_strategy.bitstring}")
    print(f"Time: {result_ts.time_taken:.2f}s")
