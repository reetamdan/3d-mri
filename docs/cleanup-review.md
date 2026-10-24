# Final cleanup review — 25 October 2026

Cleanup was performed after the real browser demo, full 187-test/CUDA suite,
typing/lint/dependency checks and final isolated wheel build passed. The project
root was checked with `git status` and a path-scoped `git clean -ndX` dry run.
Only reproducible generated files or completed one-off launch wrappers were removed.

## Removed

| Path | Checked purpose and reason |
|---|---|
| `build/` | Setuptools-generated copies of `app/`, `src/`, `scripts/` and package resources; final source/wheel tests already passed and the builder recreates them |
| `mri3d_research.egg-info/` | Generated source-build metadata; the installed editable package has separate metadata under `.venv/Lib/site-packages`; `pip check` still passed after removal |
| `.test-artifacts/` | Completed pytest fixtures, cache, earlier isolated-build/wheel outputs and a superseded cleanup-coverage database; real datasets/runs live elsewhere and the scratch parent is recreated by the test hook |
| `.mypy_cache/`, `.ruff_cache/` | Disposable checker caches; source/configurations are retained |
| `.coverage`, `coverage.xml` | Generated local coverage outputs; final counts/scope are documented, and CI/regression commands regenerate these files |
| `runs/research-serial.cmd` | One-off already-completed decoder/sanity launch sequence; both completed run trees, logs, configurations and general reproduction commands are retained |
| `runs/coordinate-sanity.cmd` | One-off completed coordinate diagnostic launcher; its outputs, source, cache lineage and documented CLI command are retained |

The two launch wrappers were read, reference-searched, and checked against the
completed studies; no training process was running when they were removed.

## Retained after review

- Application/source modules, regression tests and all scientifically distinct
  presets: these support the working app, reproduction or future comparisons.
- Dataset source/processed files, checkpoint/cache/reconstruction/generated
  artifacts, original baseline and completed research studies: these support
  inference, figure preparation and auditability.
- Failed scientific attempts and negative-result logs: failures belong in the
  research record, not the disposable category.
- `runs/PROJECT-DEMO-20261024/`: screenshots, downloaded evidence, browser report
  and executable acceptance checks support the final report and future UI checks.
- The environment, requirements variants, milestone documents and report handoff:
  these remain useful for running, installing and presenting the project.
- `runs/project-dashboard.cmd`: retained local convenience launcher for the
  verified website; the standard README launch command remains authoritative.
- `.pytest-tmp/`: an older permission-blocked scratch directory. It is ignored
  and unused; cleanup did not change its permissions or take ownership.

There is no evidence that the retained research/production files have exhausted
their purpose. Source-code removal was therefore not justified. Future launches
may recreate normal import caches; that does not mean research artifacts were
deleted or that the repository contains untracked source changes.
