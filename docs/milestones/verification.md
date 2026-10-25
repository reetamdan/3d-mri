# M1–M4 implementation record

The academic milestones cover problem analysis, planning, model design,
implementation, real-data processing and trained-model inference.

| Milestone | Material |
|---|---|
| M1 | `m1-problem-analysis.md` |
| M2 | `m2-planning-justification.md` |
| M3 | `m3-design-implementation.md` |
| M4 | `m4-demo.md`; current application and recorded model outputs |

The historical M4 run has 96 training / 24 validation exams, 360 acquired series,
128×128×32 sagittal model inputs, VQ training at 40 epochs / 1,920 updates and
diffusion training at 300 epochs / 14,400 updates. Selected epochs are 38 and 283.
Baseline reconstruction measurements are in `../results/m4-baseline.md`; the
subsequent representation study is in `../research_v2_report.md`.

Current development commands and CI are described in `../engineering-verification.md`.
The report-writing outline is `../REPORT_HANDOFF.md`. The website provides MRI
tools, measurements and terminology; academic material remains in this directory.

Research project only. Not for real medical use.
