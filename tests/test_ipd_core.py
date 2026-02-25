import pytest
from itertools import product

from ipd_core import (
    Move,
    create_strategy_from_bitstring,
    generate_random_strategy,
    IPDGame,
    TFT,
    ALLD,
)


def _idx_for_history(history, m):
    if m == 1:
        last_round = history[-1]
        return (last_round[0].value * 2 + last_round[1].value) + 1
    idx = 0
    for my_move, opp_move in history[-m:]:
        idx = idx * 4 + (my_move.value * 2 + opp_move.value)
    return idx + 1


def test_memory_mapping_bijection_for_full_histories():
    for m in [1, 2, 3, 4, 5]:
        expected_len = 5 if m == 1 else 1 + (4 ** m)
        s = generate_random_strategy(m)
        assert len(s.bitstring) == expected_len

        idxs = []
        for seq in product(range(4), repeat=m):
            history = [(Move(v // 2), Move(v % 2)) for v in seq]
            idxs.append(_idx_for_history(history, m))

        assert min(idxs) == 1
        assert max(idxs) == 4 ** m
        assert len(set(idxs)) == 4 ** m


def test_invalid_bitstring_length_raises():
    with pytest.raises(ValueError):
        create_strategy_from_bitstring('0010', memory_depth=1)
    with pytest.raises(ValueError):
        create_strategy_from_bitstring('0' * 20, memory_depth=2)


def test_round_robin_scores_are_own_scores():
    game = IPDGame()
    res = game.round_robin_tournament([TFT, ALLD], num_rounds=5, include_self_play=True)

    # TFT self-play should be 15 in 5 rounds of mutual cooperation
    assert 15 in res['TFT']['scores']
    # ALL-D self-play should be 5 in 5 rounds of mutual defection
    assert 5 in res['ALL-D']['scores']
