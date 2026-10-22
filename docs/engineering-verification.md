# Evaluation, dashboard and packaging verification

Verified locally on 22 October 2026 with Python 3.12.15, PyTorch 2.14.1+cu130,
Streamlit 1.66.1, and an NVIDIA GeForce RTX 3060 Laptop GPU.

## Results

| Check | Result |
|---|---|
| Ruff | Passed |
| Mypy | Passed across 64 runtime source files |
| Pytest | 144 passed; no local skips |
| Automated statement/branch coverage | 72.60%, including the new output auditor |
| Empty-data workspace with CUDA hidden | 138 passed, 6 expected skips; 70.52% coverage |
| Coverage denominator | `src/`, `app/`, `mri3d/`, `scripts/` |
| Required coverage floor | 70% |
| Installed requirements | `pip check`: no broken requirements |
| Built wheel | Public presets loaded; tiny CPU training/evaluation completed outside checkout |
| CUDA | Mixed precision, EMA, guidance, DDIM and VQ forward/backward passed |
| Live trained-checkpoint inference | Reconstruction and new-seed generation passed in a temporary run copy |

The expanded denominator includes orchestration/download/report scripts that were
excluded from the historical 87.82% research-v2 coverage number and the new
`scripts/verify_outputs.py` auditor. Its full real-data execution is recorded
separately from automated-suite coverage in the
[output/usefulness assessment](results/output-verification.md). The preceding
141-test hardening check had 78.17% coverage before the auditor was added. Type checking
uses `check_untyped_defs`; strict annotation completeness is not enabled.

## Regression contracts

- Overlap auditing derives processed-file hashes from actual bytes. Missing
  declared hashes cannot hide a duplicate tensor; incorrect hashes are rejected.
  Existing fingerprint calculations and checkpoint lineage remain compatible.
- Incomplete embedded legacy content references require a matching training root.
  Missing patient IDs remain explicitly exam-level.
- Frozen protocol IDs, cohorts and split are checked before model inference or
  output writes. Concurrent reservation has one winner. Completion records the
  saved report's hash; failed/interrupted attempts remain traceable and consumed.
- Persistent sample browsing validates model-pair lineage and per-sample shape,
  tolerates malformed history, and detects tensor corruption. Array cross-sections
  carry tensor-axis labels and independent per-run/sample slice state.
- Display checksum caching is invalidated by file signatures and metadata changes.
  Repeated unchanged checks perform zero additional full tensor hashes. Explicit
  reverification clears the display cache; core evaluation hashes independently.
- CPU numerical fixtures exercise M4/study stage sequencing, resume/reuse, changed
  plan/checkpoint/cache rejection, preprocessing script entry points and CLI defaults.
- The wheel test builds and installs non-editably into isolated temporary storage,
  asserts installed module locations, loads every preset, and trains/evaluates an
  artificial fixture from an unrelated working directory using isolated Python.
  No dataset/network download is part of these tests.
- Independent metric oracles check known reconstruction errors, tied-score AUC,
  missing classes and paired-bootstrap behavior. The separate real-data audit
  verifies saved tensors/latents, full-cohort metrics, repeated sampling and an
  actual three-seed augmentation comparison with a label-matched real control.

## Display performance observation

For the local 120-scan M4 dataset, one full cold dashboard fingerprint took
**0.504 seconds** and a subsequent signature-checked warm fingerprint took
**0.099 seconds** (approximately **5.1×** faster). Timing depends on the host and
filesystem cache; automated regression tests assert hashing work rather than a
machine-specific latency threshold.

## Reproduce

```bash
python -m ruff check .
python -m mypy
python -m pytest -q --cov --cov-report=term-missing
python -m pip check
```

Fresh clones explicitly skip tests requiring local MRI samples, trained checkpoints
or CUDA hardware. GitHub Actions defines Linux/Windows CPU checks using a built
installation plus isolated wheel tests; the local verification above was Windows.
Historical browser verification is recorded separately in `docs/ui-verification.md`.
