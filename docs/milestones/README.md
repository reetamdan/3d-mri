# College M1–M4 submission pack

Prepared: **21 October 2026**. Dates below are the supplied assessment windows;
this pack does not backdate implementation or claim an assessment has been passed.

**Project:** Reproducible 3D Knee MRI Generation Using Vector-Quantized Latent Diffusion.

| College assessment | Marks | Deliverable |
|---|---:|---|
| M1: Problem domain identification and analysis, Sep 7–11 | 3 | [Problem and domain analysis](m1-problem-analysis.md) |
| M2: Existing systems and feasibility, Sep 21–25 | 3 | [M2 sections 1–3](m2-planning-justification.md) |
| M2: Objectives and methodology | 3 | [M2 sections 4–6](m2-planning-justification.md) |
| M2: Algorithm/technique relevance | 3 | [M2 section 7](m2-planning-justification.md) |
| M3: Design synchronized with data handling and baseline, Oct 12–16 | 3 | [Design and implementation](m3-design-implementation.md), generated experiment evidence |
| M4: Module implementation and trained-model demo, Oct 26–30 | 3 | [Demo procedure](m4-demo.md), generated experiment evidence and actual checkpoints |

See [verification results](verification.md) for the executed checks and actual
trained-baseline results, including observed generation-quality limitations.

These college assessments are different from the repository's older **technical
Milestones 1–5** (data → VQ → diffusion → generation → downstream). Read the
college mapping above when preparing the review.

## Evidence, rather than manually entered results

The M4 runner writes `evidence.json`, `evidence.md`, training curves, and slice
comparison images into the selected experiment folder. Numbers in those files
are computed from training logs, held-out reconstruction reports, generation
reports, source manifests, and checkpoint hashes. Use those outputs for slides.
Do not substitute historical `SMOKE_TEST-*` results.

## Submission details supplied by the team

Add the actual student names/roll numbers, supervisor, department, institution,
and the institution's cover page. A prescribed page layout or report template
was not supplied. The Markdown content is organized by the actual rubric, but
institution-specific margins, signatures, and declarations must use that template.
M5 contribution statements must describe work actually performed by each member;
no team identities or contributions are invented here.

## Scientific scope

The M4 run is a reduced-cohort academic baseline with real knee MRI, disjoint
official training/validation assignments, and measured training/evaluation.
It is not a clinical model or a full MRNet benchmark. Downstream augmentation
benefit and segmentation are separate experiments, not prerequisites silently
claimed by this submission pack.
