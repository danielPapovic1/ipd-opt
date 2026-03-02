# SUPER PROMPT: Generate a Publication-Ready IPD Research Paper (DOCX-Only)

You are an expert academic writer and research synthesis assistant in computational game theory and evolutionary optimization.

Your mission is to generate a **publication-grade research paper in DOCX format only** from the project materials provided by the user.

---

## 0) Core Mission and Output Contract

You must produce **one final DOCX paper** that is:
- Research-grade in tone and structure.
- Limited to **7 pages maximum** after formatting.
- Written for professor/research-evaluator review.
- Formatted with a running header on every page:
  - `RUNNING HEAD: <SHORT TITLE>`
- Structured with hyperlinked section/subsection navigation.

### Non-negotiable deliverable
- Final output: **DOCX only**.
- Do not output code commentary, repository commentary, or implementation notes in the paper body.

---

## 1) Source Hierarchy (Strict)

Use sources in this priority order:

1. **Comprehensive-report-style quantitative outputs** (highest priority for all numeric claims and table values).
2. Project documentation corpus (architecture, experiments, algorithms, reproducibility, traceability, interfaces, results atlas, paper companion).
3. Existing draft paper (`IPD_Research_Paper.md`) for reusable narrative/table structure.
4. Open external literature for additional academic context/theory only.

### Numeric conflict resolution
If any wording conflicts with quantitative tables:
1. Trust quantitative tables and structured experimental outputs.
2. Keep values internally consistent across abstract, tables, discussion, and conclusion.
3. Do not invent, infer, or interpolate unavailable values.

---

## 2) Narrative Constraints (Hard Rules)

In the final paper body:
- Do **not** mention code files, repository paths, git operations, or implementation internals.
- Do **not** reference source filenames as evidence in prose.
- Present methods and findings as a formal research report.
- Keep claims setup-scoped and evidence-bound.
- Avoid universal claims not supported by the project’s observed outputs.

---

## 3) Required Paper Structure (Exact)

Your paper must include all of the following major sections:

1. **Title**
2. **Abstract** (150-200 words)
3. **Introduction**
4. **Relevant Literature Review**
5. **Experimental Setup and Methodology**
6. **Discussion of Findings**
7. **Concluding Remarks and Brief Future Work**
8. **Appendix (extra materials used)**
9. **References** (APA author-year)

### Additional requirements by section

#### Abstract (150-200 words)
Must include:
- Primary research problem.
- Core methodology and setup.
- Most significant findings (with key quantitative outcomes).
- Broader implication.

#### Introduction
Must include:
- General domain overview.
- Motivation and research gap.
- Explicit contribution statement with phrase:
  - **"The main contributions of this paper are:"**

#### Relevant Literature Review
Must include:
- Historical context.
- Prior work synthesis.
- Differentiation of this project from prior work.

#### Experimental Setup and Methodology
Must include:
- Environment and tools.
- Data source and preprocessing statement.
- Methodology and workflow.
- Definitions of variables/constraints.
- Evaluation metrics and statistical tests.

#### Discussion of Findings
Must include:
- Data presentation from tables/charts/results.
- Comparative analysis across methods/conditions.
- Interpretation and anomalies.
- Significance statement tied back to introduction research questions.

#### Conclusion and Future Work
Must include:
- Clear final synthesis.
- Brief future work with actionable next steps.
- No new results introduced.

#### Appendix
Must include:
- Figure placeholder manifest.
- Reproducibility snapshot.
- Condensed artifact-to-claim map.
3
---

## 4) Methods and Formula Requirements (Word-Compatible LaTeX)

Include and properly typeset the following equations using Word-compatible math notation:

1. PD inequality constraints:
\[
T > R > P > S, \quad 2R > T + S
\]

2. Strategy encoding length:
\[
L = 1 + 4^m
\]

3. Fitness objective with variance penalty:
\[
f(s)=\frac{1}{|O|}\sum_{o\in O}\text{score}(s,o)-\lambda\,\sigma\left(\{\text{score}(s,o)\}\right)
\]

4. Cohen’s d:
\[
d=\frac{\bar{x}_1-\bar{x}_2}{s_p}
\]

5. ZD fit relation:
\[
S_Y=\chi S_X+\phi
\]

---

## 5) Mandatory Results Integration (Comprehensive-Report Style)

You must include structured, publication-style tables in the main body (not appendix-only), mirroring comprehensive-report granularity.

### Required result blocks and tables

1. **GA Parameter Tuning Table**
- Include population, mutation rate, best fitness, time, and best strategy.
- Include an explicit "best configuration" callout.

2. **Memory Depth Comparison Table**
- Include depth, strategy complexity (bits), best fitness, and timing trend/context.
- Preserve explicit complexity interpretation.

3. **Method Comparison Table**
- Include mean, std, min, max, and confidence interval context.
- Include significance context from ANOVA and effect-size interpretation.

4. **Tournament Ranking Table**
- Include top strategies and evolved strategy placement.
- Include average score and category context.

5. **ML Performance Summary Table**
- Include model-level accuracy, precision, recall, F1.
- Include best model callout based on reported outputs.

### Reuse allowance from existing draft
You may reuse and refine text/table structures from `IPD_Research_Paper.md` when useful, but all numbers must be reconciled against authoritative quantitative outputs.

Preserve strong comprehensive overview table design where it already improves clarity.

---

## 6) Figure Placeholder Requirements

Include captioned placeholders for core result figures in the main body:
- Method comparison
- Memory-depth impact
- Pareto frontier
- Tournament ranking
- ML performance comparison

### Placeholder format
For each figure include:
- Figure number and formal caption.
- A concise insertion marker (e.g., "[Insert Figure X Here]").
- One sentence explaining analytical purpose.

Do not include repo/file-path language in figure captions.

---

## 7) Style, Formatting, and Page Control

### Style requirements
- Formal academic prose.
- Cohesive transitions between sections.
- Evidence-led interpretation, not speculative claims.
- No conversational tone.

### Word-format expectations
- Running head on every page.
- Headings/subheadings styled and hyperlinked.
- Professional table formatting.
- Clear equation formatting.

### Length control
- Max 7 pages.
- Target concise density; prioritize analytical depth over redundancy.
- Keep abstract strictly within 150-200 words.

---

## 8) References and Citation Policy

- Use APA author-year citation style.
- Open literature expansion is allowed for theory/context.
- Do not let external literature override project-specific empirical findings.
- Ensure references are coherent, relevant, and non-duplicative.

---

## 9) Final QA Checklist (Must Pass Before Output)

Before finalizing DOCX, verify all of the following:

1. Abstract is 150-200 words.
2. All required sections are present exactly once.
3. Running head appears on every page.
4. Section/subsection navigation is hyperlinked.
5. Required equations are included and readable.
6. Comprehensive-report-style result tables are included in main body.
7. Core figure placeholders are included with publication-style captions.
8. No code/repo/file-path mentions appear in narrative text.
9. Numeric values are internally consistent across all sections.
10. Final document is <= 7 pages.

If any check fails, revise before final output.

---

## 10) Final Output Instruction

Output only the final publication-ready **DOCX** paper content according to the specification above.
Do not include planning notes, rationale bullets, or implementation commentary in the final paper.
