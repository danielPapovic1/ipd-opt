# Spec Extract and Spec Conflicts

## Sources checked
- `README.md` (primary available specification).
- Codebase implementation files (`ipd_core.py`, `optimization.py`, `experiments.py`, `ml_prediction.py`, `main.py`).
- `project_overview.md` and `prisoners_dilemma_content`: **not found in repository tree** during audit command discovery.

## Extracted spec items
- Payoff matrix: `(R,S,T,P) = (3,0,5,1)` and matrix values `(C,C)->(3,3), (C,D)->(0,5), (D,C)->(5,0), (D,D)->(1,1)`.
- Horizon model: fixed round count (`num_rounds`, default 100), no continuation probability model.
- Noise model: none documented or implemented.
- Opponent pool in full suite includes TFT, TF2T, STFT plus ALLD, ALLC, GRIM, PAVLOV.
- Baselines include TFT, TF2T, STFT, ALLD, ALLC.
- Optimizers implemented: GA, EDA, Hill Climbing, Tabu Search.
- Memory depths 1..5 are intended in experiments.
- ML predictor exists with multiple sklearn models.

## Conflicts detected
1. **Bitstring length spec conflict**
   - README specifies memory-1 encoding with 5 bits and memory-n as `1 + 4^n`.
   - Prompt/spec requirement asks deterministic memory-m bitstrings of length exactly `4^m`.
   - Implementation follows README-style `1 + 4^m` (except hardcoded 5 at m=1).

2. **Canonical ordering only explicitly defined for m=1**
   - README gives `(CC, CD, DC, DD)` for m=1.
   - No explicit multi-round ordering/indexing spec in docs.
   - Implementation has an indexing formula for m>1, but it collapses to a single index due clipping behavior.

3. **Tournament score bookkeeping mismatch**
   - Tournament result structure suggests each strategy stores its own score list.
   - Implementation appends opponent score into each strategy's `scores` list.

4. **README reproducibility gap**
   - README lists run commands but omits environment setup details (virtualenv/OS expectations, expected runtime, deterministic seed instructions).

