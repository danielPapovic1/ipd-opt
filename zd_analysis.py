"""
Zero-Determinant (ZD) Strategy Analysis
======================================
Approximate diagnostics for detecting linear payoff enforcement in IPD strategies.
"""

from typing import List, Tuple, Dict
import numpy as np

from ipd_core import Strategy, IPDGame, REFERENCE_STRATEGIES


def is_zero_determinant(
    strategy: Strategy,
    opponents: List[Strategy],
    num_rounds: int = 100
) -> Tuple[bool, float, float]:
    """
    Approximate whether a strategy enforces a linear relationship:
        alpha * S_X + beta * S_Y + gamma ~= 0
    by fitting:
        S_Y = chi * S_X + phi

    Returns:
        (is_zd, chi, phi)
    """
    game = IPDGame()
    sx_vals: List[float] = []
    sy_vals: List[float] = []

    for opponent in opponents:
        sx, sy, _ = game.play_match(strategy, opponent, num_rounds)
        sx_vals.append(float(sx) / float(num_rounds))
        sy_vals.append(float(sy) / float(num_rounds))

    x = np.array(sx_vals, dtype=float)
    y = np.array(sy_vals, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]):
        return False, 0.0, 0.0

    # Least squares fit y = chi * x + phi
    design = np.column_stack([x, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    chi = float(coeffs[0])
    phi = float(coeffs[1])

    y_hat = chi * x + phi
    residual = float(np.mean((y - y_hat) ** 2))
    denom = float(np.var(y)) + 1e-12
    r2 = float(1.0 - (np.var(y - y_hat) / denom))

    # Practical threshold for near-linear enforcement.
    is_zd = (r2 >= 0.98) and (residual <= 0.02)
    return is_zd, chi, phi


def analyze_zd_properties(strategy_list: List[Strategy]) -> List[Dict]:
    """
    Analyze and classify strategies by approximate ZD behavior.
    Classification:
      - Extortionate: chi > 1
      - Generous: chi < 1
      - Non-ZD: not approximately linear
    """
    results: List[Dict] = []
    opponents = REFERENCE_STRATEGIES

    for strategy in strategy_list:
        is_zd, chi, phi = is_zero_determinant(strategy, opponents)
        if not is_zd:
            zd_type = "Non-ZD"
        elif chi > 1.0:
            zd_type = "Extortionate"
        elif chi < 1.0:
            zd_type = "Generous"
        else:
            zd_type = "Non-ZD"

        results.append({
            'strategy': strategy.name,
            'bitstring': strategy.bitstring,
            'is_zd': bool(is_zd),
            'chi': float(chi),
            'phi': float(phi),
            'classification': zd_type
        })

    return results
