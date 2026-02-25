# Execution Log (Commands + Outcomes)

## Environment
- `uname -a`, `python --version`, `python -m pip --version`, `lscpu | head -n 20`

## Dependency install
- `python -m pip install -r requirements.txt`
- Outcome: success (requirements already satisfied).

## README commands
- `python main.py --demo` → success; produced expected demo output including optimization + ML sections.
- `python main.py --report` → success; regenerated `results/comprehensive_report.txt`.
- `python main.py --full` → started but did not complete in practical audit window; process manually terminated after extended runtime.

## Tests / checks run
- `pytest -q` → no tests discovered.
- Custom inline Python audit scripts executed for:
  - memory mapping bijection checks across m=1..5,
  - baseline strategy behavior micro-matches,
  - optimizer smoke tests at m=1,3,5,
  - deterministic rerun check under fixed seeds.

## Important observed outputs
- Memory mapping check showed unique index count of **1** for m=2..5 (non-bijective/collapsed mapping).
- Optimizer smoke runs succeeded at m=1,3,5; produced bitstrings of lengths 5, 65, and 1025 respectively.

