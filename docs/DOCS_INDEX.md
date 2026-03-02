# IPD Documentation Index

This folder is the technical documentation entrypoint for the Iterated Prisoner's Dilemma (IPD) optimization project.

Primary audiences:
- Graders and supervisors evaluating methodology, rigor, and reproducibility.
- Maintainers extending algorithms, interfaces, and experiment pipelines.
- Research-paper authors extracting evidence-backed claims from code and artifacts.

## Reading Paths

### 15-minute grader path
1. [EXPERIMENTS.md](./EXPERIMENTS.md)
2. [RESULTS_ATLAS.md](./RESULTS_ATLAS.md)
3. [TRACEABILITY_MATRIX.md](./TRACEABILITY_MATRIX.md) (Key Findings section)

### 45-minute maintainer path
1. [ARCHITECTURE.md](./ARCHITECTURE.md)
2. [INTERFACES_AND_DATA_CONTRACTS.md](./INTERFACES_AND_DATA_CONTRACTS.md)
3. [ALGORITHMS_DEEP_DIVE.md](./ALGORITHMS_DEEP_DIVE.md)
4. [REPRODUCIBILITY_AND_PROVENANCE.md](./REPRODUCIBILITY_AND_PROVENANCE.md)

### Research-paper author path
1. [PAPER_COMPANION.md](./PAPER_COMPANION.md)
2. [EXPERIMENTS.md](./EXPERIMENTS.md)
3. [RESULTS_ATLAS.md](./RESULTS_ATLAS.md)
4. [TRACEABILITY_MATRIX.md](./TRACEABILITY_MATRIX.md)
5. [ARCHITECTURE.md](./ARCHITECTURE.md)

## Documentation Map

- [ARCHITECTURE.md](./ARCHITECTURE.md): Runtime architecture, execution flows, module boundaries, complexity, and extension points.
- [EXPERIMENTS.md](./EXPERIMENTS.md): Protocols, hyperparameters, reproducibility policy, artifact schemas, and interpretation boundaries.
- [INTERFACES_AND_DATA_CONTRACTS.md](./INTERFACES_AND_DATA_CONTRACTS.md): Canonical structures, function I/O contracts, invariants, and compatibility rules.
- [ALGORITHMS_DEEP_DIVE.md](./ALGORITHMS_DEEP_DIVE.md): GA/EDA/HillClimbing/TabuSearch internals, objective variants, and tuning implications.
- [RESULTS_ATLAS.md](./RESULTS_ATLAS.md): File-by-file interpretation of all outputs in `results/`, including plot reading guidance.
- [REPRODUCIBILITY_AND_PROVENANCE.md](./REPRODUCIBILITY_AND_PROVENANCE.md): Metadata interpretation, seed policy, deterministic vs variable outputs, and provenance chains.
- [TRACEABILITY_MATRIX.md](./TRACEABILITY_MATRIX.md): Claim-level mapping from docs/report statements to code lines and result artifacts.
- [PAPER_COMPANION.md](./PAPER_COMPANION.md): Structured paper-writing template and evidence placement guide.

## Quick Links to Evidence

Source of intent:
- [repo-blueprint/ipd_repo_blueprint.md](../repo-blueprint/ipd_repo_blueprint.md)

Primary run report:
- [results/comprehensive_report.txt](../results/comprehensive_report.txt)

Core result tables:
- [results/ga_parameter_tuning.csv](../results/ga_parameter_tuning.csv)
- [results/memory_depth_results.csv](../results/memory_depth_results.csv)
- [results/method_comparison.csv](../results/method_comparison.csv)
- [results/tournament_results.csv](../results/tournament_results.csv)
- [results/ml_results.csv](../results/ml_results.csv)
- [results/pareto_frontier.csv](../results/pareto_frontier.csv)
- [results/statistical_analysis.json](../results/statistical_analysis.json)
- [results/zd_analysis.json](../results/zd_analysis.json)
- [results/reproducibility_metadata.json](../results/reproducibility_metadata.json)

Key plots:
- [results/ga_parameter_heatmap.png](../results/ga_parameter_heatmap.png)
- [results/memory_depth_impact.png](../results/memory_depth_impact.png)
- [results/method_comparison.png](../results/method_comparison.png)
- [results/tournament_results.png](../results/tournament_results.png)
- [results/evolved_comparison.png](../results/evolved_comparison.png)
- [results/ml_comparison.png](../results/ml_comparison.png)
- [results/pareto_frontier.png](../results/pareto_frontier.png)
- [results/convergence_ci.png](../results/convergence_ci.png)
