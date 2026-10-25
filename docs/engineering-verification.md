# Development checks

The project uses Ruff, mypy, pytest, package-install checks and Linux/Windows CI.
The automated suite covers tensor shapes, forward/backward execution, checkpoint
resume, seeded inference, geometry, content hashes, model/data relationships,
viewer state, sample controls and export workflows.

```bash
python -m ruff check .
python -m mypy
python -m pytest -q --cov --cov-report=term-missing
python -m pip check
```

The complete functional run before the final wording/cleanup pass completed
193 tests with 73.71% statement/branch coverage. Current source checks are
recorded by the repository's [CI workflow](https://github.com/reetamdan/3d-mri/actions/workflows/tests.yml).

The application was exercised with Chrome on the local MRI/model artifacts:
native navigation, linked images, all recorded model variants, image downloads,
live inference, raw-series controls, NIfTI processing/export and responsive
layouts at 900, 390 and 320 px. Temporary browser audit tools and their disposable
copies were removed after use. The automated regression suite remains part of
normal development.

Research project only. Not for real medical use.
