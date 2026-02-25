"""
Iterated Prisoner's Dilemma (IPD) Core Module
=============================================
Core game engine, payoff matrix, and tournament system.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum

class Move(Enum):
    """Possible moves in Prisoner's Dilemma"""
    COOPERATE = 0
    DEFECT = 1
    
    def __repr__(self):
        return "C" if self == Move.COOPERATE else "D"

# Standard IPD Payoff Matrix (R, S, T, P)
# R = Reward for mutual cooperation = 3
# S = Sucker's payoff = 0
# T = Temptation to defect = 5
# P = Punishment for mutual defection = 1
PAYOFF_MATRIX = {
    (Move.COOPERATE, Move.COOPERATE): (3, 3),   # R, R
    (Move.COOPERATE, Move.DEFECT): (0, 5),      # S, T
    (Move.DEFECT, Move.COOPERATE): (5, 0),      # T, S
    (Move.DEFECT, Move.DEFECT): (1, 1),         # P, P
}

@dataclass
class Strategy:
    """Represents a strategy for playing IPD"""
    name: str
    play_func: Callable
    is_bitstring: bool = False
    bitstring: Optional[str] = None
    memory_depth: int = 1
    
    def play(self, history: List[Tuple[Move, Move]]) -> Move:
        """Make a move based on game history"""
        return self.play_func(history)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Strategy):
            return self.name == other.name
        return False

class IPDGame:
    """Iterated Prisoner's Dilemma Game Engine"""
    
    def __init__(self, payoff_matrix: Dict = PAYOFF_MATRIX):
        self.payoff_matrix = payoff_matrix
    
    def play_round(self, move1: Move, move2: Move) -> Tuple[int, int]:
        """Play a single round and return payoffs"""
        return self.payoff_matrix[(move1, move2)]
    
    def play_match(self, strategy1: Strategy, strategy2: Strategy, 
                   num_rounds: int = 100) -> Tuple[int, int, List[Tuple[Move, Move]]]:
        """
        Play a match between two strategies for num_rounds.
        Returns (score1, score2, history)
        """
        history = []
        score1, score2 = 0, 0
        
        for _ in range(num_rounds):
            # Each strategy makes a move based on history
            move1 = strategy1.play(history)
            move2 = strategy2.play([(h[1], h[0]) for h in history])  # Flip perspective
            
            # Get payoffs
            payoff1, payoff2 = self.play_round(move1, move2)
            score1 += payoff1
            score2 += payoff2
            
            # Record history
            history.append((move1, move2))
        
        return score1, score2, history
    
    def round_robin_tournament(self, strategies: List[Strategy], 
                                num_rounds: int = 100,
                                include_self_play: bool = True) -> Dict[str, Dict]:
        """
        Run a round-robin tournament where each strategy plays against every other.
        Returns dictionary with results for each strategy.
        """
        results = {s.name: {'total_score': 0, 'matches': 0, 'scores': []} 
                   for s in strategies}
        
        n = len(strategies)
        for i in range(n):
            for j in range(i if include_self_play else i + 1, n):
                score_i, score_j, _ = self.play_match(
                    strategies[i], strategies[j], num_rounds
                )
                
                # Update results
                results[strategies[i].name]['total_score'] += score_i
                results[strategies[i].name]['matches'] += 1
                results[strategies[i].name]['scores'].append(score_j)
                
                results[strategies[j].name]['total_score'] += score_j
                results[strategies[j].name]['matches'] += 1
                results[strategies[j].name]['scores'].append(score_i)
        
        # Calculate average scores
        for name in results:
            if results[name]['matches'] > 0:
                results[name]['avg_score'] = results[name]['total_score'] / results[name]['matches']
            else:
                results[name]['avg_score'] = 0
        
        return results
    
    def get_payoff_matrix_table(self) -> np.ndarray:
        """Return payoff matrix as numpy array for visualization"""
        return np.array([
            [3, 0],
            [5, 1]
        ])


# ============== REFERENCE STRATEGIES ==============

def tit_for_tat(history: List[Tuple[Move, Move]]) -> Move:
    """
    Tit-for-Tat: Cooperate on first move, then copy opponent's last move.
    Nice, Provokable, Forgiving, Clear.
    """
    if not history:
        return Move.COOPERATE
    return history[-1][1]  # Opponent's last move

def tit_for_two_tats(history: List[Tuple[Move, Move]]) -> Move:
    """
    Tit-for-Two-Tats: Cooperate first two moves, defect only after 
    two consecutive opponent defections.
    More forgiving than TFT.
    """
    if len(history) < 2:
        return Move.COOPERATE
    # Check if opponent defected twice in a row
    if history[-1][1] == Move.DEFECT and history[-2][1] == Move.DEFECT:
        return Move.DEFECT
    return Move.COOPERATE

def suspicious_tit_for_tat(history: List[Tuple[Move, Move]]) -> Move:
    """
    Suspicious Tit-for-Tat: Defect on first move, then copy opponent's last move.
    """
    if not history:
        return Move.DEFECT
    return history[-1][1]

def always_defect(history: List[Tuple[Move, Move]]) -> Move:
    """ALL-D: Always defect. Dominant strategy in single-shot PD."""
    return Move.DEFECT

def always_cooperate(history: List[Tuple[Move, Move]]) -> Move:
    """ALL-C: Always cooperate. Can be exploited."""
    return Move.COOPERATE

def random_strategy(history: List[Tuple[Move, Move]], 
                   p_cooperate: float = 0.5) -> Move:
    """RAND: Randomly cooperate with probability p_cooperate."""
    return Move.COOPERATE if random.random() < p_cooperate else Move.DEFECT

def grim_trigger(history: List[Tuple[Move, Move]]) -> Move:
    """
    Grim Trigger: Cooperate until opponent defects once, then always defect.
    """
    for round_hist in history:
        if round_hist[1] == Move.DEFECT:
            return Move.DEFECT
    return Move.COOPERATE

def pavlov(history: List[Tuple[Move, Move]]) -> Move:
    """
    Pavlov (Win-Stay, Lose-Shift): Cooperate if previous round was mutual 
    cooperation or mutual defection. Defect otherwise.
    """
    if not history:
        return Move.COOPERATE
    my_last, opp_last = history[-1]
    if my_last == opp_last:
        return Move.COOPERATE
    return Move.DEFECT

# Create strategy instances
TFT = Strategy("TFT", tit_for_tat, memory_depth=1)
TF2T = Strategy("TF2T", tit_for_two_tats, memory_depth=2)
STFT = Strategy("STFT", suspicious_tit_for_tat, memory_depth=1)
ALLD = Strategy("ALL-D", always_defect, memory_depth=0)
ALLC = Strategy("ALL-C", always_cooperate, memory_depth=0)
RAND = Strategy("RAND", lambda h: random_strategy(h, 0.5), memory_depth=0)
GRIM = Strategy("GRIM", grim_trigger, memory_depth=float('inf'))
PAVLOV = Strategy("PAVLOV", pavlov, memory_depth=1)

REFERENCE_STRATEGIES = [TFT, TF2T, STFT, ALLD, ALLC, RAND, GRIM, PAVLOV]


def create_strategy_from_bitstring(bitstring: str, memory_depth: int = 1) -> Strategy:
    """
    Create a strategy from a bitstring representation.
    
    For memory_depth=1: bitstring has 5 bits
        - bit 0: initial move (0=C, 1=D)
        - bits 1-4: response to (C,C), (C,D), (D,C), (D,D)
    
    For memory_depth=n: bitstring has 1 + 2^(2n) bits
    """
    def bitstring_strategy(history: List[Tuple[Move, Move]]) -> Move:
        if not history:
            # First move
            return Move.DEFECT if bitstring[0] == '1' else Move.COOPERATE
        
        # Encode last 'memory_depth' rounds as index
        if memory_depth == 1:
            last_round = history[-1]
            # Encode: (my_move, opp_move) -> 0, 1, 2, 3
            idx = (last_round[0].value * 2 + last_round[1].value) + 1
        else:
            # For deeper memory, encode all rounds
            idx = 1
            for i in range(max(0, len(history) - memory_depth), len(history)):
                round_moves = history[i]
                idx = idx * 4 + (round_moves[0].value * 2 + round_moves[1].value)
            idx = min(idx, len(bitstring) - 1)
        
        return Move.DEFECT if bitstring[idx] == '1' else Move.COOPERATE
    
    return Strategy(
        name=f"Bitstring_{bitstring[:10]}...",
        play_func=bitstring_strategy,
        is_bitstring=True,
        bitstring=bitstring,
        memory_depth=memory_depth
    )


def generate_random_strategy(memory_depth: int = 1) -> Strategy:
    """Generate a random strategy bitstring"""
    if memory_depth == 1:
        length = 5  # 1 initial + 4 responses
    else:
        length = 1 + (4 ** memory_depth)
    
    bitstring = ''.join(random.choice('01') for _ in range(length))
    return create_strategy_from_bitstring(bitstring, memory_depth)


if __name__ == "__main__":
    # Test the game engine
    game = IPDGame()
    
    print("=" * 60)
    print("IPD Game Engine Test")
    print("=" * 60)
    
    # Test individual matches
    print("\n1. Testing Individual Matches (100 rounds each):")
    print("-" * 50)
    
    test_pairs = [
        (TFT, ALLC, "TFT vs ALL-C"),
        (TFT, ALLD, "TFT vs ALL-D"),
        (TFT, TFT, "TFT vs TFT"),
        (ALLD, ALLC, "ALL-D vs ALL-C"),
        (TF2T, ALLD, "TF2T vs ALL-D"),
    ]
    
    for s1, s2, name in test_pairs:
        score1, score2, _ = game.play_match(s1, s2, 100)
        print(f"{name:20s}: {s1.name}={score1:4d}, {s2.name}={score2:4d}")
    
    # Test tournament
    print("\n2. Round-Robin Tournament:")
    print("-" * 50)
    
    strategies = [TFT, TF2T, ALLD, ALLC, RAND]
    results = game.round_robin_tournament(strategies, 100)
    
    # Sort by average score
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['avg_score'], 
                           reverse=True)
    
    print(f"{'Rank':<6}{'Strategy':<15}{'Avg Score':<12}{'Total Score':<12}")
    print("-" * 50)
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"{rank:<6}{name:<15}{data['avg_score']:<12.2f}{data['total_score']:<12}")
    
    print("\n3. Bitstring Strategy Test:")
    print("-" * 50)
    
    # Create a TFT-like bitstring strategy
    # Initial: C (0), then: (C,C)->C, (C,D)->D, (D,C)->C, (D,D)->D
    tft_bits = "00101"  # C, C, D, C, D
    tft_clone = create_strategy_from_bitstring(tft_bits, 1)
    
    score1, score2, _ = game.play_match(tft_clone, TFT, 100)
    print(f"TFT-clone vs TFT: {score1}, {score2} (should be ~300, 300)")
