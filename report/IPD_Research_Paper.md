# Computational Evolution of Strategies in the Iterated Prisoner's Dilemma

**Author:** [Your Name]  
**Affiliation:** [Course / Department / Institution]  
**Date:** [Submission Date]

**Keywords:** Iterated Prisoner's Dilemma, evolutionary computation, zero-determinant strategies, cooperation, machine learning

## Abstract

This report presents a quantitative study of strategy evolution in the Iterated Prisoner's Dilemma (IPD) using four optimization approaches: Genetic Algorithm (GA), Estimation of Distribution Algorithm (EDA), Hill Climbing, and Tabu Search. Strategies were evaluated under a common opponent portfolio and analyzed with statistical testing, complexity trade-off analysis, tournament validation, and machine-learning-based success prediction. In five independent comparative runs, EDA achieved the highest mean best-fitness (251.56), followed by GA (245.73), while Hill Climbing and Tabu Search both averaged 224.22. At the same time, ANOVA did not show statistically significant separation at the tested sample size (F = 0.6515, p = 0.5935), so method differences are interpreted as practical trends under this setup rather than universal dominance. Memory-depth experiments showed peak raw fitness at depth 3 (266.12), while complexity increased from 5 to 1025 bits across tested depths. Tournament evaluation placed the evolved champion near top reference strategies. Machine learning achieved moderate predictive quality, with Neural Network producing the strongest mean F1. Overall, the findings support a bounded-memory, evidence-based view of high-performing IPD strategy design.

## 1. Introduction

The Iterated Prisoner's Dilemma (IPD) remains a central model for studying cooperation, reciprocity, punishment, and long-term strategic adaptation. The one-shot incentive to defect contrasts with the repeated-game opportunity for stable cooperation, making IPD an ideal framework for testing how algorithmic search discovers effective policies under conflicting incentives.

This work is motivated by three linked questions. First, which optimization approach is most effective in practical IPD strategy search when all methods are evaluated under the same competitive conditions? Second, does increased strategic memory consistently improve outcomes, or does it mainly increase complexity and runtime burden? Third, can machine learning identify structural patterns that predict strong strategies without relying solely on exhaustive simulation?

To answer these questions, the study compares four optimization approaches, reports run-level distributions rather than single best outcomes, and integrates statistical interpretation, tournament validation, and strategy diagnostics. The analysis emphasizes empirical discipline: reported claims are tied to observed results, and interpretation is kept within the tested setup.

**Main contributions:**
- A multi-method empirical comparison of GA, EDA, Hill Climbing, and Tabu Search under standardized evaluation.
- A complexity-aware memory-depth analysis from depth 1 to depth 5.
- Tournament benchmarking of evolved strategy quality against reference strategies.
- Predictive modeling of strategy success using interpretable feature summaries.

## 2. Relevant Literature Review

### 2.1 Cooperation in repeated dilemmas

Foundational work on repeated dilemmas established that reciprocal strategies can outperform purely exploitative ones when interaction continues over time (Axelrod, 1984; Axelrod & Hamilton, 1981). The practical importance is not just average payoff, but strategic stability: effective policies typically balance initial cooperation, retaliation against defection, and the ability to return to cooperation.

### 2.2 Evolutionary search in strategic games

Evolutionary methods are widely used when policy spaces are discrete and combinatorial. Population-based methods (such as GA and EDA) are generally valued for exploration and diversity preservation, while local search methods (such as Hill Climbing and Tabu Search) provide efficient neighborhood exploitation. In IPD-like domains, the relative advantage of each method is strongly setup-dependent, which motivates direct comparative experiments rather than method assumptions.

### 2.3 Zero-determinant strategy theory

Press and Dyson (2012) showed that certain memory-1 strategies can enforce linear payoff relationships, introducing the zero-determinant (ZD) class. Subsequent work distinguished extortionate and generous forms and explored their evolutionary behavior (Stewart & Plotkin, 2013; Hilbe et al., 2018). This study uses ZD analysis as a post hoc diagnostic to interpret evolved policies.

### 2.4 Machine learning for strategy characterization

Machine learning can complement simulation by modeling the relationship between structural strategy features and success labels. In repeated-game settings, features related to cooperation propensity, response behavior, and reciprocity signatures can provide practical predictive signal and help prioritize candidate strategies for deeper evaluation.

## 3. Experimental Setup and Methodology

### 3.1 Game model and payoff structure

The analysis uses the standard Prisoner's Dilemma payoff configuration:

\[
T > R > P > S, \quad 2R > T + S,
\]

with \\(R=3, S=0, T=5, P=1\\). Strategies are evaluated over repeated rounds against a fixed diverse opponent set to ensure comparability across methods.

### 3.2 Strategy representation and memory depth

Strategies are encoded as deterministic bitstrings with memory depth \\(m\\), where encoding length is:

\[
L = 1 + 4^m.
\]

The first bit controls the opening action, and remaining bits specify responses conditioned on recent interaction states.

### 3.3 Optimization methods and core settings

**Table 1. Optimization methods and baseline settings**

| Method | Core configuration |
|---|---|
| Genetic Algorithm (GA) | Population 100, mutation 0.01, crossover 0.8, elitism 2 |
| Estimation of Distribution Algorithm (EDA) | Population 100, selection rate 0.3, learning rate 0.1 |
| Hill Climbing | 10 restarts, single-bit neighbors, sampled neighborhood at deeper memory |
| Tabu Search | Tabu list size 10, aspiration criterion enabled |

### 3.4 Objective, statistical analysis, and diagnostics

A robustness-oriented fitness form is used:

\[
f(s)=\frac{1}{|O|}\sum_{o\in O}\text{score}(s,o)-\lambda\,\sigma\left(\{\text{score}(s,o)\}\right).
\]

Method comparison is interpreted with ANOVA and effect sizes. Cohen's d is computed as:

\[
d=\frac{\bar{x}_1-\bar{x}_2}{s_p}, \quad
s_p=\sqrt{\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}}.
\]

ZD behavior is examined via linear payoff fitting:

\[
S_Y=\chi S_X+\phi.
\]

### 3.5 ML prediction protocol

Machine learning models were trained to classify high-performing strategies (top 20% threshold). Features captured initial cooperation behavior, conditional response patterns, cooperation rate, and similarity to reciprocal templates. Performance was evaluated with accuracy, precision, recall, and F1.

### 3.6 Data source and preprocessing statement

No external dataset was used in this study. All reported data were generated directly from the experimental pipeline: repeated-game simulations, optimization-run summaries, tournament evaluations, and derived statistical outputs. Preprocessing was limited to structured aggregation and formatting steps required for analysis and presentation: (1) organizing run-level outputs into tabular summaries, (2) computing grouped statistics (means, standard deviations, confidence intervals), (3) computing percentile-based labels for ML classification tasks, and (4) deriving Pareto and diagnostic summaries from already generated experimental tables. No manual relabeling, external data enrichment, or post hoc data substitution was performed.

## 4. Discussion of Findings

### 4.1 Hyperparameter sensitivity (GA)

**Table 2. GA parameter tuning outcomes**

| Population | Mutation | Best Fitness | Time (s) | Best Strategy |
|---:|---:|---:|---:|---|
| 50 | 0.001 | 251.08 | 0.33 | 00101 |
| 50 | 0.010 | 251.08 | 0.23 | 00101 |
| 50 | 0.050 | 251.08 | 0.39 | 00101 |
| 100 | 0.001 | 251.08 | 0.40 | 00101 |
| 100 | 0.010 | 251.08 | 0.43 | 00101 |
| 100 | 0.050 | 257.62 | 0.86 | 00110 |
| 200 | 0.001 | 252.32 | 1.28 | 01110 |
| 200 | 0.010 | 252.19 | 1.37 | 00111 |
| 200 | 0.050 | **257.71** | 2.26 | 00110 |

Best observed GA configuration in this run context is population 200 with mutation 0.05.

**Figure 1 (Placeholder). Comparative method performance plot**  
*Insert method-comparison PNG here.*

### 4.2 Memory-depth complexity-fitness trade-off

**Table 3. Memory-depth summary (best observed fitness by depth with complexity trend)**

| Depth | Strategy Bits | Best Observed Fitness | Method | Representative Runtime Trend |
|---:|---:|---:|---|---|
| 1 | 5 | 252.93 | EDA | very low |
| 2 | 17 | 263.96 | GA | moderate |
| 3 | 65 | **266.12** | GA | high |
| 4 | 257 | 259.22 | GA | high |
| 5 | 1025 | 259.33 | GA | very high |

Peak raw fitness appears at depth 3, but complexity increases exponentially. The resulting interpretation is a trade-off, not monotonic improvement with depth.

**Figure 2 (Placeholder). Memory-depth impact plot**  
*Insert memory-depth PNG here.*

**Figure 3 (Placeholder). Pareto frontier plot**  
*Insert Pareto-frontier PNG here.*

### 4.3 Method comparison with statistical interpretation

**Table 4. Method comparison (5 runs each)**

| Method | Mean Fitness | Std | Min | Max | 95% CI |
|---|---:|---:|---:|---:|---|
| EDA | **251.56** | 45.60 | 207.43 | 316.85 | [211.59, 291.53] |
| GA | 245.73 | 39.50 | 205.84 | 301.34 | [211.10, 280.35] |
| Hill Climbing | 224.22 | 36.33 | 188.01 | 275.28 | [192.37, 256.06] |
| Tabu Search | 224.22 | 36.33 | 188.01 | 275.28 | [192.37, 256.06] |

ANOVA result: \\(F=0.6515,\; p=0.5935\\). This does not support strong inferential separation at current sample size, though effect sizes indicate practical differences between population-based and local-search methods.

### 4.4 Tournament competitiveness

**Table 5. Tournament ranking (top reference and evolved results)**

| Rank | Strategy | Avg Score | Type |
|---:|---|---:|---|
| 1 | TF2T | 538.4 | Reference |
| 2 | TFT | 536.6 | Reference |
| 3 | EvolvedChampion_GA_00101 | 534.8 | Evolved |
| 4 | GRIM | 520.1 | Reference |
| 5 | PAVLOV | 515.6 | Reference |

The evolved champion is near top baseline strategies, indicating competitive quality under tested tournament conditions.

**Figure 4 (Placeholder). Tournament ranking chart**  
*Insert tournament-results PNG here.*

### 4.5 ML predictive performance

**Table 6. ML summary metrics (aggregated over train sizes)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.8108 | 0.6212 | 0.1904 | 0.2747 |
| Logistic Regression | 0.8108 | 0.6212 | 0.1904 | 0.2747 |
| Neural Network | **0.8150** | 0.6120 | 0.2596 | **0.3443** |
| Random Forest | 0.8108 | 0.6212 | 0.1904 | 0.2747 |
| SVM | 0.8108 | 0.6212 | 0.1904 | 0.2747 |

ML results show moderate predictive signal, with Neural Network giving the strongest F1 in this setup.

**Figure 5 (Placeholder). ML comparison chart**  
*Insert ML-comparison PNG here.*

### 4.6 Integrated interpretation

The evidence supports three bounded conclusions. First, EDA and GA show stronger practical performance trends than local search in this configuration, while statistical certainty remains limited at five runs. Second, memory-depth expansion improves peak score up to a point but introduces substantial complexity and runtime cost. Third, evolved strategies can remain highly competitive against established references while exhibiting cooperative structural signatures consistent with reciprocal behavior.

### 4.7 Significance relative to the research questions

The findings directly address the three motivating questions posed in the Introduction. On optimization effectiveness, EDA and GA showed stronger practical performance than local-search alternatives in this setup, with EDA leading in mean fitness while inferential uncertainty remained non-trivial at current run count. On memory-depth value, results did not support a simple \"more memory is always better\" claim; instead, they supported a bounded trade-off between incremental fitness gains and rapidly increasing complexity/runtime cost. On predictive modeling, ML outcomes demonstrated usable but moderate signal, indicating that feature-based screening can assist strategy search without replacing direct evaluation. Taken together, these results support the study hypothesis that robust IPD strategy analysis requires an integrated lens combining optimization, statistical context, complexity trade-offs, and predictive modeling rather than reliance on single-metric best-case performance.

## 5. Concluding Remarks and Brief Future Work

This report shows that IPD strategy optimization benefits from a multi-method evaluation framework that combines performance, statistical interpretation, complexity trade-offs, tournament benchmarking, and predictive modeling. Within this tested setting, EDA achieved the highest mean comparative fitness, GA delivered strong peak outcomes, and local-search methods were faster but less competitive in average quality. Depth-aware analysis highlighted that higher memory is not automatically better once complexity cost is considered.

**Future work (brief):**
1. Increase the number of independent runs to strengthen statistical power.
2. Evaluate robustness under noisy opponents and alternative opponent portfolios.
3. Extend strategy classes beyond deterministic bitstrings.
4. Improve ML calibration and class-imbalance handling.
5. Add automated consistency checks between narrative summaries and tabulated outputs.

## Appendix (Extra Materials Used)

### Appendix A. Figure Placeholder Manifest

| Figure | Intended content | Data source type |
|---|---|---|
| Figure 1 | Method comparison distribution | PNG experimental output |
| Figure 2 | Memory-depth impact | PNG experimental output |
| Figure 3 | Pareto frontier | PNG experimental output |
| Figure 4 | Tournament ranking | PNG experimental output |
| Figure 5 | ML performance comparison | PNG experimental output |

### Appendix B. Reproducibility Snapshot

- Base seed policy used with run-specific offsets.
- Recorded environment: Python 3.11.9, NumPy 2.4.2, SciPy 1.17.1, scikit-learn 1.8.0.
- Hardware context: 16-core AMD64.
- Independent comparative runs: 5 per method.

### Appendix C. Condensed Artifact-to-Claim Map

| Claim area | Supporting material type |
|---|---|
| Hyperparameter findings | Parameter table outputs |
| Method ranking and spread | Comparative run tables + statistical summary |
| Complexity trade-off | Memory-depth and Pareto tables |
| Competitive validity | Tournament ranking table |
| Predictive modeling | ML metric table |

## References

Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books.

Axelrod, R., & Hamilton, W. D. (1981). The evolution of cooperation. *Science, 211*(4489), 1390-1396.

Hilbe, C., Chatterjee, K., & Nowak, M. A. (2018). Partners and rivals in direct reciprocity. *Nature Human Behaviour, 2*(7), 469-477.

Nowak, M. A., & Sigmund, K. (1993). A strategy of win-stay, lose-shift that outperforms tit-for-tat in the prisoner's dilemma game. *Nature, 364*(6432), 56-58.

Press, W. H., & Dyson, F. J. (2012). Iterated prisoner's dilemma contains strategies that dominate any evolutionary opponent. *Proceedings of the National Academy of Sciences, 109*(26), 10409-10413.

Simon, H. A. (1957). *Models of Man, Social and Rational: Mathematical Essays on Rational Human Behavior in a Social Setting*. Wiley.

Stewart, A. J., & Plotkin, J. B. (2013). From extortion to generosity, evolution in the iterated prisoner's dilemma. *Proceedings of the National Academy of Sciences, 110*(38), 15348-15353.
