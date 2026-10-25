# Project cleanup record

The completed one-off browser/inspection tools and their temporary copies are
removed after use. They are separate from the application's ongoing research,
training and automated CI code.

## Reviewed removals

- Completed one-off tools under `runs/UI-OVERHAUL-20261025/`,
  `runs/PROJECT-DEMO-20261024/` and the earlier output study's `audit-tools/`.
- Temporary checkpoint copies, browser downloads and standalone UI captures from
  those one-off checks.
- `scripts/verify_outputs.py` and its dedicated numerical-audit test file; the
  completed numerical observations are retained in `docs/results/output-verification.md`.
- `scripts/smoke_research.py`, an earlier standalone numerical smoke runner.
- Superseded standalone browser/compliance check documents, consolidated into
  the current README and development notes.
- Generated checker, build, test-fixture and coverage outputs when their current
  execution is complete.

## Ongoing files

Application modules, model code, configuration presets, the automated CI
regression suite and general research/measurement commands remain in use.
Original datasets, trained weights, experiment records and study figures remain
with their local `data/` and `runs/` folders. The installed environment and the
local dashboard launcher support the running application.

An older permission-restricted `.pytest-tmp/` directory is ignored and is not
used by the current test configuration. Its ownership/settings were not changed.

Research project only. Not for real medical use.
