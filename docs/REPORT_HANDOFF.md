# Project report writing handoff

Last updated: 25 October 2026. This is the working handoff for the final project
report, presentation and viva. Update the completion/verification section when
the dashboard and final GitHub publication are verified.

## Project in one paragraph

This project implements a reproducible 3D knee MRI research workflow: inspect and
preprocess real MRI, learn a compact representation, reconstruct scans, train and
sample conditional latent diffusion, and evaluate reconstruction, generation and
possible classification augmentation. A controlled study on 96 training and 24
model-selection validation exams found that a compact MONAI KL autoencoder with
less slice-direction compression improves reconstruction over the existing VQ
model. Fine-detail and tiny-data diffusion gates still fail, so realistic knee
generation, verified pathology conditioning and beneficial synthetic augmentation
are not achieved claims.

## Master research scope and evidence status

| Work | Actual status | Evidence |
|---|---|---|
| Baseline/source audit and preservation | Complete | `docs/current_baseline.md`; original M4 checkpoint hashes preserved |
| Data and orientation validation | Complete for array-grid properties; patient-space metadata unavailable | `runs/REPRESENTATION-DATA-AUDIT/`; `docs/research_v2_report.md` |
| Representation bake-off | Complete: 15 matched-budget runs, nine variants | `docs/results/ae_bakeoff_measured.md` and `.json` |
| Larger-data pretraining | Adapter and exposure auditing implemented/tested; external pretraining not run | `src/data/knee_pretraining.py`; compatible authorized data required |
| Latent analysis | Complete for selected frozen KL cache; generic VQ/KL tools available | `runs/DIFFUSION-SANITY-02/full-cache/manifest.json` |
| Diffusion/conditioning sanity | Artificial constant conditioning passed; four one-MRI diagnostics failed | `runs/DIFFUSION-SANITY-02/`, `runs/DIFFUSION-COORD/` |
| Full-cohort improved diffusion, CFG/conditioning sweeps | Components/configs implemented; serious runs gated/deferred | Representation detail and one-volume sanity gates did not pass |
| Medical features and memorization | Completed for six primary reconstructions and eight historical generations | `runs/AE-FEATURE-AUDIT/`; pinned MedicalNet provenance |
| Strong conventional/downstream comparison | Framework tested; serious new synthetic comparison not run | `src/training/downstream_fair.py`; earlier negative probe below |
| Project dashboard/browser verification | Implemented and browser-verified | `app/project_results.py`; `runs/PROJECT-DEMO-20261024/browser-report.json`; `docs/PROJECT_DEMO.md` |
| Final cleanup and GitHub publication | Reviewed cleanup complete; source commits and live GitHub checks linked below | `docs/cleanup-review.md`; research evidence preserved |

The original research scope is not reduced to a successful-generation claim.
Unmet gates and unavailable data are recorded explicitly rather than running
costly unsupported experiments merely to fill every stage.

## Report chapter map

1. **Introduction:** problem, objectives, educational/research scope, constrained GPU.
2. **Background and related work:** representation learning, VQ/KL, latent
   diffusion, conditioning; use the papers and inspected classes in
   `docs/external_repos_review.md`.
3. **Data and preprocessing:** official 96/24 split, 128×128×32 array grid,
   trilinear resizing, volume min-max [-1,1], missing physical/patient metadata.
4. **Methodology:** architecture map, isotropic versus anisotropic KL, losses,
   training-only weight calibration, same 1,920-update/effective-batch-two budget,
   deterministic reconstruction, recorded seeds and checkpoint selection.
5. **Implementation:** PyTorch/MONAI, config-driven experiments, frozen latent
   caches, overlap/content auditing, Streamlit viewers, testing and packaging.
6. **Results:** full autoencoder table; paired exam intervals and seed SD; negative
   perceptual/decoder results; failed diffusion sanity and earlier augmentation.
7. **Discussion:** evidence for a representation bottleneck, compression/detail
   tradeoff, noise sensitivity of edge proxies, why metrics do not prove pathology.
8. **Conclusion and future work:** measured reconstruction improvement; compatible
   pretraining, independent anatomy evaluation, then gated generative experiments.

## Numbers to use, with correct comparisons

### Matched representation controls — three seeds (42,43,44)

| Model | PSNR dB, mean ± seed SD | SSIM, mean ± seed SD |
|---|---:|---:|
| Matched native VQ | 24.352 ± 0.100 | 0.7691 ± 0.0081 |
| KL isotropic | 25.502 ± 0.070 | 0.8169 ± 0.0037 |
| KL anisotropic | 26.802 ± 0.199 | 0.8578 ± 0.0092 |

Anisotropic KL improves mean PSNR by 2.451 dB and SSIM by .0887 versus
the **new matched VQ**. These are internal validation results. All representations
fail the frozen gradient/high-frequency detail gate.

The historical M4 VQ is a different checkpoint: PSNR 24.29968, SSIM .76587.
Interpolation is 21.36067 / .77707. Do not mix the historical score with the
three-seed matched control when quoting an improvement.

The exploratory lower-compression KL reaches 29.07243 / .91578 on one seed,
but eight latent channels at 2×2×2 reduction provide **no float-scalar compression**.
Its gradient ratio .570 and HF ratio .255 still fail the gate. Treat this as a
tradeoff, not the main compressed-latent winner.

### Tiny-data diffusion — intentionally fitting one training MRI

| Diagnostic, 2,000 updates each | SSIM to training reconstruction | Normalized latent MSE | Decision |
|---|---:|---:|---|
| Unweighted v | .63788 | .32813 | Failed |
| Min-SNR gamma 5 | .53915 | .41431 | Failed |
| Array coordinates | .66138 | .30715 | Failed |
| Coordinates + Min-SNR | .57944 | .37355 | Failed |

The fixed gate is SSIM ≥.9 and latent MSE ≤.1. Four-volume runs were skipped.
The artificial ±.5 conditioning control passed; this is software sanity, not
evidence of generated ACL/meniscus lesions.

### Earlier augmentation probe — three seeds, 400 updates per arm

| Task, mean validation AUROC | Real only | Real + synthetic | Label-matched resampling |
|---|---:|---:|---:|
| Abnormal | .7614 | .5860 | .7684 |
| ACL | .2517 | .4219 | .2704 |
| Meniscus | .2786 | .3405 | .2857 |

At threshold .5, ACL and meniscus sensitivity were zero. This is a negative,
limited probe, not a useful pathology classifier. The stronger TorchIO/Youden/
five-seed framework is implemented but has no completed new utility result.

## Figure inventory and captions

| Figure | Source | Suggested caption |
|---|---|---|
| System workflow | Streamlit methodology / `docs/current_baseline.md` | Real knee MRI to learned representation, reconstruction and experimental latent diffusion |
| Data audit | `runs/REPRESENTATION-DATA-AUDIT/` | Source/processed array-grid views and intensity/slice-count distributions |
| Matched reconstruction comparisons | `runs/AE-BAKEOFF/comparison-*.png`, `errors-*.png` | Identical validation cases and slice positions across interpolation, VQ and KL |
| Compression tradeoff | `runs/AE-LOW-COMPRESSION/comparison-*.png` | Higher detail at a larger continuous representation, not equivalent compression |
| Three-seed model comparison | Published measured table / Streamlit experiment chart | Internal validation mean ± training-seed SD at matched update budgets |
| Diffusion failure | `runs/DIFFUSION-COORD/overfit-1-baseline.png` | Samples fail to reproduce even a single intentionally fitted training reconstruction |
| Nearest neighbors | `runs/AE-FEATURE-AUDIT/neighbors-*.png` | Historical generation versus all-training medical-feature and SSIM neighbors |
| Final application | Browser screenshots recorded after verification | The tested project dashboard with synchronized real reconstruction comparisons |

Use the same slice positions and intensity range. Include median/worst/random
cases, not only the best image. Central grid crops are not annotated ACL ROIs.

## Claims supported and not supported

**Supported:** functioning reproducible software; measured reconstruction gain
across three seeds; evidence that representation/compression choice matters;
bounded negative generative results; conservative gate-based compute discipline.

**Not established:** verified preservation of ACL tears/meniscus pathology,
patient-independent generalization, realistic new full-knee synthesis, reliable
pathology conditioning, or useful synthetic-data augmentation. No aligned
pathology masks, independent test cohort or expert review were available.

## Reproduction and viva

- Full research commands: `docs/research_v2_report.md`.
- External sources, licenses and papers: `docs/external_repos_review.md`.
- Test/build evidence: `docs/engineering-verification.md`.
- Launch: `python -m streamlit run app/streamlit_app.py`.
- Demo sequence: overview → identical-slice reconstruction comparison → experiment
  results → optional live reconstruction → clearly labeled experimental generation.
- Detailed current walkthrough: `docs/PROJECT_DEMO.md`. One-line technical
  explanations are available in the Streamlit sidebar, metric help and glossary.
- Explain **reconstruction versus generation** first: an improved reconstruction
  is not a newly generated knee, and label requests are not verified pathology.

## Verification and publication log

- Research source commits: `e2facc6`, `3207cbc` on `research/ae-ldm-v2`.
- Before the UI pass: 184 full local tests, 72.88% coverage; clean-data/CUDA-hidden
  checkout 178 passed/6 expected skips, 71.42%; mypy 81 files, Ruff and isolated
  wheel passed. Hosted Linux/Windows CI succeeded (run `40034983420`).
- Final UI/browser checks, final test counts, cleanup decisions and final GitHub
  commit/CI references are maintained below as completion work finishes.

### Completed website verification

Chrome **156.0.8078.47** exercised the actual localhost application: verified real
MRI comparisons; all three confirmation seeds; best/median/worst/random cases;
array-axis/slice navigation; interpolation/error views; live trained KL inference;
CSV and handoff downloads; preserved diffusion navigation; 900 and 390 px layouts.
Observed JavaScript exceptions, console errors and failed requests: **zero**.
Initial live inference/model load took 28.46 s; a cached-model call took .49 s.
Screenshots were inspected; they show actual model outputs without sharpening.
Full browser evidence and executable local checks are retained under
`runs/PROJECT-DEMO-20261024/` for figures and later verification.

### Final automated project checks

The complete available Windows/CUDA suite passed **187 tests** with **73.36%**
statement/branch coverage in 203.80 s. Mypy checked **83 source files**; Ruff,
dependency consistency and isolated non-editable wheel/preset tests passed.
See `docs/results/project-demo-verification.md`. Browser checks are separate
evidence, not included in automated-suite coverage.
A separate empty-data checkout of source `9202dbe`, with CUDA hidden, passed
**181 tests / six expected skips / 72.02% coverage**, in 125.21 s. The final
dashboard source also passed a separate PEP 517 isolated build.

### Cleanup

After the working demo and final checks passed, reviewed generated build/test/
checker/coverage outputs and two completed one-off launch wrappers were removed.
`docs/cleanup-review.md` records each decision. Production/research code, tests,
datasets, models, useful figures and negative results remain available.

### Versioned source and GitHub references

- Completed dashboard/report source: commit `9202dbe`; subsequent documentation
  commits record clean-checkout verification and finalization.
- Main project: <https://github.com/reetamdan/3d-mri/tree/main>.
- Research history: <https://github.com/reetamdan/3d-mri/tree/research/ae-ldm-v2>.
- Current hosted checks: <https://github.com/reetamdan/3d-mri/actions/workflows/tests.yml>.
- Read the exact uploaded revision with `git rev-parse origin/main`; read current
  CI with `gh run list --branch main --workflow tests.yml --limit 1`.

These links identify the versioned source and current CI rather than freezing a
potentially stale badge result into a report draft. Cite the commit/check run
used when submitting the final report. MRI datasets, weights and MRI screenshots
stay in the documented local artifact folders rather than the source upload.
