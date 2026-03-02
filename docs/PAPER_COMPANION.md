# Paper Companion

This guide maps repository evidence to a structured research-paper workflow.

Primary goal:
- Enable direct drafting of a rigorous paper without re-reading source code.

Related docs:
- [DOCS_INDEX.md](./DOCS_INDEX.md)
- [EXPERIMENTS.md](./EXPERIMENTS.md)
- [RESULTS_ATLAS.md](./RESULTS_ATLAS.md)
- [TRACEABILITY_MATRIX.md](./TRACEABILITY_MATRIX.md)

## 1. Recommended Paper Structure

## 1.1 Abstract
Include:
- Problem: evolving effective IPD strategies.
- Method summary: GA, EDA, HC, TS + ML prediction + statistical analysis.
- Main outcomes: best configuration/method metrics and tournament placement.

Evidence sources:
- `results/comprehensive_report.txt` (sections 3 and 4)
- `results/method_comparison.csv`
- `results/tournament_results.csv`

## 1.2 Introduction
Include:
- Why IPD is relevant to cooperative behavior modeling.
- Why comparing multiple optimization paradigms is useful.
- Why ML-based success prediction is included.

Evidence sources:
- `README.md`
- `docs/ARCHITECTURE.md`
- `repo-blueprint/ipd_repo_blueprint.md`

## 1.3 Methodology
Subsections:
1. Game environment and payoff matrix.
2. Strategy encoding and memory depth.
3. Optimization algorithms and objective function.
4. Experiment protocols and controls.
5. Statistical and structural analyses.

Evidence sources:
- `ipd_core.py:26-31`, `ipd_core.py:210-257`
- `optimization.py:40-170`, `optimization.py:175-756`
- `experiments.py:101-499`
- `analysis.py:32-126`, `zd_analysis.py:13-86`

## 1.4 Results
Subsections:
1. Hyperparameter sensitivity (GA tuning).
2. Memory depth trade-offs.
3. Method-level comparison with statistical context.
4. Tournament performance of evolved champion vs references.
5. ML predictive performance.

Evidence sources:
- `results/ga_parameter_tuning.csv`
- `results/memory_depth_results.csv`
- `results/method_comparison.csv`
- `results/statistical_analysis.json`
- `results/tournament_results.csv`
- `results/ml_results.csv`

## 1.5 Discussion
Include:
- Interpretation boundaries.
- What current results imply for algorithm selection.
- What current results imply for depth complexity trade-offs.

Evidence sources:
- `docs/EXPERIMENTS.md` section "Interpretation Boundaries"
- `docs/RESULTS_ATLAS.md`

## 1.6 Conclusion
Include:
- Evidence-backed summary of practical findings.
- Practical recommendations for future extensions.

Evidence sources:
- Aggregated means from `results/*.csv`
- `docs/ALGORITHMS_DEEP_DIVE.md`

## 2. Section-by-Section Evidence Placement

| Paper section | Minimum required evidence |
|---|---|
| Abstract | At least 3 numeric results from CSV/JSON artifacts |
| Methods | At least 1 code anchor per method subsection |
| Results | Every claim backed by a table or JSON metric |
| Discussion | Distinguish observation vs interpretation explicitly |
| Conclusion | No new claims that did not appear in results |

## 3. Reusable Figure and Table Set

## 3.1 Recommended figures
1. `results/ga_parameter_heatmap.png`
2. `results/memory_depth_impact.png`
3. `results/method_comparison.png`
4. `results/tournament_results.png`
5. `results/ml_comparison.png`
6. `results/pareto_frontier.png`
7. `results/convergence_ci.png`

## 3.2 Recommended core tables
1. GA hyperparameter best rows from `ga_parameter_tuning.csv`.
2. Memory depth summary with `best_fitness`, `time_taken`, `strategy_bits`.
3. Method comparison summary (`mean,std,min,max`) from `method_comparison.csv`.
4. Top tournament ranks from `tournament_results.csv`.
5. ML metric means by model from `ml_results.csv`.
6. ANOVA + key effect sizes from `statistical_analysis.json`.

## 4. Claim Wording Templates by Evidence Strength

## 4.1 Directly observed numeric claims
Template:
- "Under the configured run policy, [method/config] achieved [value] in [artifact]."

Example:
- "Under the configured run policy, GA with population 200 and mutation 0.05 achieved best fitness 257.7147 in `ga_parameter_tuning.csv`."

## 4.2 Comparative claims with statistical context
Template:
- "[Method A] showed higher mean performance than [Method B] in this run set; however, ANOVA p-value was [value], so this should be treated as configuration-specific evidence."

## 4.3 Diagnostic classification claims
Template:
- "Strategies were classified as approximately [Generous/Extortionate/Non-ZD] by the linear-fit diagnostic (`zd_analysis.py`)."

## 4.4 Scope-limited generalization claims
Template:
- "These results indicate [trend] within the evaluated opponent set and objective settings."

## 5. Writing Do/Do-Not Rules

Do:
1. Cite artifact files directly for all numeric values.
2. Use code line anchors for methodological claims.
3. Keep run context explicit (seed policy, rounds range, train sizes).

Do not:
1. Claim universal algorithm superiority from a single run family.
2. Treat approximate ZD diagnostics as symbolic proofs.
3. Use narrative report text when it conflicts with raw artifact values.

## 6. Drafting Checklist for Paper Consistency

1. Every numeric claim has a direct artifact citation.
2. Every method claim has a code anchor.
3. Terminology is consistent: fitness, rounds, memory depth, strategy bits.
4. Reported best model/method aligns with current CSV/JSON values.
5. File naming is accurate (for example, `convergence_ci.png` not `convergence_comparison.png`).
6. Conclusions do not introduce unsupported extrapolation.
7. References across sections are non-contradictory.

## 7. Suggested Citation Bundle for Appendix

Include these in technical appendix:
- `docs/TRACEABILITY_MATRIX.md`
- `results/statistical_analysis.json`
- `results/reproducibility_metadata.json`
- `docs/INTERFACES_AND_DATA_CONTRACTS.md`

This bundle supports high-auditability grading and reproducible interpretation.
