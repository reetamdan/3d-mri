# Completed project dashboard verification

Verified 24–25 October 2026 on the local Windows / RTX 3060 Laptop environment.
The application uses the existing trained models; this presentation pass ran no
additional representation or diffusion training.

## Automated checks

| Check | Result |
|---|---|
| Full available pytest suite, including real MRI and CUDA | **187 passed**, 203.80 seconds |
| Statement/branch coverage over `src/`, `app/`, `mri3d/`, `scripts/` | **73.36%**, required floor remains 70% |
| Mypy | **83 runtime source files**, passed |
| Ruff | Passed |
| Installed dependency consistency | `pip check` passed |
| Non-editable wheel / presets outside source checkout | Passed in the full suite |

The new tests verify model/data/report binding, changed checkpoint/image
rejection, safe original-format study compatibility, exclusion of artificial or
unfinished experiments, synchronized axis controls, comparison options and
missing-local-artifact behavior. Existing NIfTI workflow tests explicitly navigate
from the new project landing page; their prior assertions remain intact.

## Actual browser acceptance

Browser: **Chrome 156.0.8078.47**, automated with Playwright 1.64.1 using the
installed Chrome binary. Desktop viewport: 1560×1400; responsive checks: 900×1000
and 390×1000. These are viewport checks, not physical-device certification.

- New landing page displays measured benchmark cards and verified real MRI.
- Original, native VQ and selected KL show the same exam/slice with fixed contrast.
- All three confirmation seeds load their corresponding native VQ comparison.
- Best, median, worst, seeded random and all-exam selection work.
- Array-axis selection and synchronized slice navigation work.
- Interpolation reference and fixed-range absolute-error maps render.
- Live trained KL reconstruction completes on the selected real input.
- Experiment chart, compact table, detailed measurements and CSV download work.
- Method cards, one-line glossary and report-handoff download work.
- Preserved diffusion workspace and return navigation work.
- Narrow layouts have no observed horizontal page overflow or app exception.

Observed browser JavaScript exceptions: **0**. Console errors: **0**. Failed
requests: **0**. Screenshots were read and inspected, not just produced.

First live model load/inference took **28.46 s**; a cached-model call took
**0.49 s**, with selected-case whole-volume PSNR **27.03 dB** / SSIM **.862**.
The initial model import/device setup is included in the first measurement.
Viewing saved comparisons uses cached files and does not start GPU inference.

Machine-readable browser evidence, screenshots, downloads and the executable
acceptance scripts are retained at `runs/PROJECT-DEMO-20261024/`. These local
screenshots contain MRI and remain with local data/run artifacts.

## Presentation and report entry points

- `docs/PROJECT_DEMO.md`: launch and five-minute demonstration.
- `docs/REPORT_HANDOFF.md`: living report outline, result tables, figure locations,
  captions, scientific limitations and completion/publication status.
- `docs/research_v2_report.md`: full measured research study.
- `docs/external_repos_review.md`: source-level external review and citations.

The completed deliverable is a reproducible project and verified presentation of
its measured reconstruction improvement and negative generative findings. It is
not a claim that the failed generation/anatomy/augmentation gates have passed.
