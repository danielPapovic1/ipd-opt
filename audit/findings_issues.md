# Key Issues (Defects / Risks)

## P0 correctness issues
1. **Memory-depth mapping broken for m>1**
   - For memory depths >1, strategy indexing computes `idx = idx*4 + state` starting from `idx=1`, then clamps with `min(idx, len(bitstring)-1)`.
   - For full-length histories, this maps all states to the final index (`4^m`) instead of a bijection over `1..4^m`.
   - Effect: almost entire bitstring is unreachable; behavior is effectively one response bit + initial bit.

2. **Round-robin score list stores opponent scores**
   - In `round_robin_tournament`, each strategy appends the opponent's match score to its own `scores` array.
   - Any downstream use of `scores` as self-performance is incorrect.

## P1 reproducibility / rigor issues
1. **No explicit fitness-evaluation budget accounting** across optimizers.
2. **Inconsistent evaluation conditions in multi-run comparison**:
   - rounds are randomized per run, which complicates fair cross-method comparison.
3. **No train/val/test separation for strategy generation process** in ML pipeline; one random split only.
4. **No duplicate/near-duplicate leakage controls** in ML dataset creation.
5. **README full suite runtime not bounded/documented**, and command can run for long periods without progress output in non-tty mode.

## P2 quality issues
1. No built-in unit test suite (pytest discovers zero tests).
2. No explicit validation guard that provided bitstrings match required length for selected memory depth.
3. EDA lacks probability clipping safeguards against degeneracy (0/1 collapse).

