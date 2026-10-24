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

## Representation-study extension — 24 October 2026

The earlier table records the engineering baseline before the optional MONAI/KL
research extension. The expanded source retains the same coverage denominator
and required 70% statement/branch floor.

| Check | Current extension result |
|---|---|
| Full local Windows/CUDA pytest | **184 passed**, no skips; 195.00 seconds |
| Fresh detached checkout, no local MRI/weights, CUDA hidden | **178 passed, 6 expected skips**; 107.49 seconds; **71.42% coverage** |
| Automated statement/branch coverage | **72.88%**, 5,453 statements and 1,810 branches included |
| Mypy | **81 runtime source files**, passed |
| Ruff | Passed |
| Dependency consistency | `pip check`: no broken requirements |
| PEP 517 isolated build | Passed; all build dependencies installed in isolation |
| Non-editable wheel regression | Passed; every one of 23 presets loads outside the checkout |
| CUDA and preserved live baseline | Existing mixed-precision/EMA/guidance/DDIM and temporary-copy inference tests passed |

The new regression coverage includes exact CPU representation resume, native
MONAI iso/aniso/deconvolution shapes and input gradients, train-only latent
statistics/inverse scaling, pretraining-exposure overlap rejection, target-specific
Min-SNR mathematics, null-token conditioning, matched coordinate initialization,
TorchIO determinism, tied AUPRC/operating thresholds, label-stratum coverage and
quality-gate rejection, stale feature/model lineage, and exclusion of incomplete
or artificial study tables. Numerical fixtures stay explicitly artificial.

The six emitted warnings concern PyTorch's deprecated TorchScript APIs used for
explicit frozen-feature compatibility. They are not test failures. Real model
training/audit measurements are recorded separately in
`docs/research_v2_report.md`; running a research experiment is not counted as
automated test coverage.

The clean-checkout check used source commit `e2facc6`, a separate detached
worktree with no copied data/run outputs, and the same installed research-enabled
Python environment with `CUDA_VISIBLE_DEVICES=''`. It verifies offline data/run
independence locally; the Linux/Windows hosted jobs additionally install the CPU
PyTorch wheel from scratch.

## Completed project presentation — 25 October 2026

The final dashboard extension passed **187 tests** with **73.36% coverage**;
mypy passed **83 runtime files**, Ruff and `pip check` passed. Existing wheel,
CUDA, real-scan and live-baseline cases remain in that suite. The coverage
denominator and 70% floor are unchanged.
A separate empty-data checkout of source `9202dbe`, with CUDA hidden, passed
**181 tests / six expected skips / 72.02% coverage**. A final PEP 517 isolated
wheel build also passed.

Real Chrome acceptance verified the new project landing page, original/VQ/KL
comparison, seeds/cases/axes/slices, interpolation/errors, live trained KL
reconstruction, result/report downloads, preserved prototype navigation, and
900/390 px layouts. Browser exceptions, console errors and failed requests were
zero. First-load versus cached inference were measured separately.

See `docs/results/project-demo-verification.md` for full scope and local evidence,
`docs/PROJECT_DEMO.md` for the demonstration and `docs/REPORT_HANDOFF.md` for
the maintained report-writing handoff. These browser executions are separate
from pytest coverage.
