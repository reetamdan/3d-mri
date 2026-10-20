# Repository Analysis & Milestone 2 Compliance Audit

> **Historical audit (October 6).** Current college M1–M4 evidence is in
> `docs/milestones/` and the run referenced by `runs/latest_m4.json`.
> CUDA support, real MRNet subset training, optional bottleneck attention,
> generation-evaluation tests, and live trained-model demo checks were added on
> October 21. Historical smoke artifacts lack newer integrity fields; the
> original claims below must not be treated as current artifact verification.

**Project:** 3D MRI Research Pipeline (`3d-mri`)
**Report date:** 2026-10-06
**Scope:** Full repository review (architecture, code quality, tests, application) plus a
requirement-by-requirement audit against the *Milestone 2 Review: Review of Existing System,
Feasibility, Objectives, Methodology & Algorithm Relevance* document (source basis: *Medical
Diffusion: Denoising Diffusion Probabilistic Models for 3D Medical Image Generation*, Khader et al.).

**Method:** static code review with file/line citations, execution of the full test suite,
and inspection of persisted run artifacts. No code was changed for this audit.

---

## 1. Executive summary

**Verdict — the project is fully working as a software system and substantively matches the
Milestone 2 methodology.**

- All **43 tests pass** (`43 passed in 29.56s`, executed for this audit). The README still
  states 39 — the count is stale, not the suite.
- The complete Milestone 2 chain — **VQ-GAN/VQ-VAE → latent cache → 3D DDPM → generation →
  evaluation → downstream augmentation comparison** — has been executed end-to-end twice as
  smoke runs (`runs/SMOKE_TEST-20260912-172156`, `runs/SMOKE_TEST-20260912-193057`) with full
  artifact lineage (checksums, manifests, reports) verified on disk.
- The two-stage architecture reproduces the Medical Diffusion design faithfully, including the
  paper's **compression factor of 4** (256×256×32 → 64×64×8 latent) and codebook quantization
  of generated latents before decoding.
- **Two architectural deviations** from the milestone document exist: the 3D U-Net has
  **no spatial/depth attention** (the one technique in the milestone's relevance table that is
  absent), and MRNet preprocessing defaults to **resize to 128×128×32** instead of the paper's
  center-crop to 256×256×32. Both are configurable or documented; the attention omission is the
  only substantive technical gap.
- **Three external blockers** prevent research results (as opposed to software function):
  MRNet dataset access (gated, application outstanding), a CPU-only PyTorch installation on a
  machine with an RTX 5070 Ti, and anatomical validity review (smoke scale only, by design).
- The codebase shows unusual engineering discipline for a student research project: SHA-256
  provenance chaining across every pipeline stage, genuine reproducibility engineering, and
  scientific honesty enforced in code (smoke artifacts structurally excluded from research
  paths; every report carries explicit validity caveats).

---

## 2. Verification performed

| Check | Result |
|---|---|
| Test suite (`python -m pytest -q`) | **43 passed in 29.56s** |
| Smoke run artifacts (`runs/SMOKE_TEST-20260912-193057`) | Complete chain present: `vq/`, `latents/`, `diffusion/`, `generated/`, `reconstruction/`, `generation-report.json`, `summary.json` |
| Smoke summary claims | `"status": "SMOKE TEST ONLY"`; `"mrnet_training": "Blocked: approved MRNet files unavailable"`; `"downstream_results": "Not run on brain samples: no genuine MRNet labels"` |
| Codebase size | ~2,700 lines of Python across 35 files (26 project modules, 9 test files) |
| Version control | **Not a git repository** (a `.gitignore` exists and is correct, but nothing is tracked) |
| PyTorch install | CPU-only wheel (`requirements-lock.txt` pins `torch==2.14.0`; confirmed in `docs/milestones-2-5.md`) |
| Application | Covered by Streamlit `AppTest` flows in `tests/test_app.py` and `tests/test_real_scans.py` (real-data tier skips on clean clones); dashboard binds to loopback per `.streamlit/config.toml` |

The smoke run is honest about its scale: 32³ volumes, 2 epochs, 8 diffusion steps,
`smoke_test: true` throughout. Its summary explicitly declines every claim the software cannot
support ("do not interpret output as realistic MRI").

---

## 3. System overview and architecture

### 3.1 Module map

```
scripts/ (CLI entry) ──► src/training/ ──► src/models/ (vqvae.py, diffusion.py)
        │                     │  ▲
        ▼                     ▼  │
app/ (Streamlit) ──► src/data/ ──► src/utils/ (config, logging, seed, integrity)
                          │
                          ▼
                 src/visualization/, src/metrics.py
```

- **`src/data/`** — `loader.py` (NIfTI load + validation gates), `validation.py`
  (`VolumeValidationError` codes), `preprocessing.py` (RAS canonicalization → unit-aware
  resampling → crop/pad with affine correction → endpoint-preserving resize → minmax/z-score
  normalization; every spatial op updates the affine), `indexing.py` (subject discovery,
  SHA-256 pseudonyms), `splitting.py` (seeded patient-level largest-remainder split),
  `pipeline.py` (immutable run directory with metadata.csv / experiment.json / report.json),
  `mrnet.py` (MRNet adapter preserving official train/valid splits), `dataset.py`
  (`MRIDataset` with checksum caching and cross-split leakage detection).
- **`src/models/`** — `vqvae.py`: `VQVAE3D` (two stride-2 conv downsamples → 1×1 conv to
  latent channels; chunked nearest-neighbor `VectorQuantizer` with straight-through estimator;
  mirrored transposed-conv decoder with tanh output) and `PatchDiscriminator3D`. `diffusion.py`:
  `UNet3D` (single-downsample, FiLM-style time/label conditioning via `TimeBlock`) and `DDPM`
  (float64 cosine β-schedule, ε-prediction, seeded ancestral sampling).
- **`src/training/`** — `experiments.py` is the staged workflow (`train_vq` → `cache_latents` →
  `train_diffusion` → `generate` → `evaluate_reconstruction`); `evaluation.py` (adaptive 3D
  MS-SSIM, unpaired generation diagnostics, Dice/IoU); `downstream.py` (matched-seed
  augmentation comparison with a hand-rolled Mann-Whitney AUC); `common.py` (checkpoint I/O,
  dataset fingerprinting, strict model-config validation).
- **`app/`** — `streamlit_app.py` (8-page dashboard: Dataset Overview, MRI Viewer,
  Preprocessing, Intensity Analysis, Dataset Quality, Export, MRNet Dataset, Model
  Experiments) + `research_workspace.py` (MRNet browsing, experiment dashboards with
  per-rerun integrity re-verification).

### 3.2 Defining design decision: integrity chaining

Every artifact carries a SHA-256 and every consuming stage re-verifies the lineage of its
inputs: the VQ checkpoint records the dataset fingerprint; the latent cache records the VQ
checkpoint hash; the diffusion checkpoint embeds the latent manifest; generation verifies the
VQ↔diffusion chain; every evaluation re-checks provenance before reporting. Supporting
mechanisms: `confined_path` traversal guard (`src/utils/integrity.py:6-11`), atomic checkpoint
writes (`.partial` + replace), `weights_only=True` on all `torch.load`, `allow_pickle=False`
on all `np.save`/`np.load`, and `fresh_output` refusing silent overwrites.

### 3.3 Reproducibility

Per-epoch-seeded DataLoader shuffles, fixed validation-noise RNG, seeded per-sample DDPM
sampling, full Python/NumPy/Torch/CUDA RNG state save/restore on resume, cudnn deterministic
mode, and dataset fingerprints covering both metadata and tensor bytes. Tests assert sampling
determinism directly (`tests/test_research.py:60-62`).

---

## 4. Milestone 2 compliance audit

### 4.1 Section 1 — Review of Existing System

A literature-positioning section; no code is required. The repository's framing (2D-only
approaches miss inter-slice context; GAN training instability/mode collapse; direct 3D
diffusion cost; two-stage latent diffusion as remedy) is consistent with the implementation's
design choices and with the caveats in `docs/milestones-2-5.md`.

### 4.2 Section 2 — Feasibility

| Milestone document statement | Repository status |
|---|---|
| Staged development: preprocessing → VQ-GAN/VQ-VAE → latent diffusion | Implemented exactly as proposed; each stage is a separate CLI command with persisted lineage |
| Latent-space operation reduces diffusion dimensionality | Encoders downsample ×4 per axis; diffusion trains on 16-channel latents, not voxels (`cache_latents` + `train_diffusion`) |
| Reference benchmark: Quadro RTX 6000 (24 GB), ~7 days per model | Local machine has an RTX 5070 Ti Laptop (~12 GB), **but the installed PyTorch wheel is CPU-only** — real-scale training is blocked until a CUDA wheel is installed. `configs/models.yaml` (100 epochs × 100 steps) is a research recipe, not a demonstrated run; the docs say so explicitly |
| Compression factor 4 retains anatomy better for 256×256×32 | **Matches the implementation**: two stride-2 convs give exactly ×4 per axis (256→64, 32→8) |

**Feasibility verdict:** technically feasible and demonstrably functional at smoke scale;
full-scale feasibility is gated on the CUDA wheel and dataset access, not on design risk.

### 4.3 Section 3 — Objectives

| # | Objective | Status | Evidence |
|---|---|---|---|
| 1 | Reproducible pipeline for synthetic 3D MRI | ✅ Met (software) | Seeds, configs, fingerprints, immutable run dirs, CLI with `--resume` |
| 2 | Compact latent representation via VQ-GAN/VQ-VAE | ✅ Met | `VQVAE3D` + optional patch-discriminator VQ-GAN (`gan_weight: 0.05`) |
| 3 | 3D diffusion over MRI latents | ✅ Met | `train_diffusion` on cached latents with train-only normalization |
| 4 | Generate latents from random noise | ✅ Met | `DDPM.sample` — seeded, correct posterior mean/variance |
| 5 | Decode latents into synthetic volumes | ✅ Met | `generate()` denormalizes, **snaps to codebook (`quantize=True`, `experiments.py:265`)**, decodes |
| 6 | Preserve anatomy / inter-slice consistency | ⚠️ Tooling met, result unproven | Continuity metrics, paired reconstruction viewer, MS-SSIM exist; actual preservation requires real training + expert review — reports say exactly this |
| 7 | Evaluate via quality/similarity/diversity/downstream measures | ✅ Met | PSNR/SSIM/MS-SSIM/MSE/MAE; diversity, adjacent-slice MAE, memorization screen; AUC/accuracy/Brier |
| 8 | Investigate synthetic data for augmentation | ✅ Met (tooling) | `compare_augmentation`: matched seeds, identical update budgets, cohort-leakage gates |

### 4.4 Section 4 — Methodology steps

| Step | Status | Notes |
|---|---|---|
| 1. MRNet dataset preparation | ⚠️ Adapter complete; dataset blocked | Official train/valid splits preserved, headerless label CSVs strictly validated, planes treated as independent series, no invented affines or patient IDs. MRNet access still pending (Redivis application) |
| 2. Preprocessing | ⚠️ Deviation (documented) | Adapter **resizes** to a configurable multiple-of-4 shape (default 128×128×32) rather than **center-cropping to 256×256×32** as in the reference pipeline. 256×256×32 is selectable (within the 64M-voxel guard). Docs acknowledge resizing "can still distort resolution" |
| 3. VQ-GAN/VQ-VAE encoding | ✅ Matches | ×4 compression; nearest-neighbor codebook, codebook + commitment loss, straight-through estimator |
| 4. Latent diffusion training | ✅ Core matches; ⚠️ deviations | Cosine β-schedule, ε-prediction MSE, 3D U-Net, conditional labels. **Deviations:** no spatial/depth attention; no EMA, DDIM, or classifier-free guidance; no pretrained perceptual loss — all stated in `docs/milestones-2-5.md` as compact-baseline choices |
| 5. Synthetic generation | ✅ Matches | Seeded reverse denoising from Gaussian noise; nonfinite outputs rejected |
| 6. Decoding | ✅ Matches | Latents denormalized and quantized through the learned codebook before the decoder runs |
| 7. Evaluation | ✅ Matches | Paired SSIM/PSNR for VQ reconstructions only (deliberately not for unrelated generated/real pairs); unpaired diversity/continuity/memorization diagnostics; downstream classification comparison |

### 4.5 Section 5 — Algorithm/technique relevance table

| Technique (milestone doc) | In repository | Where |
|---|---|---|
| VQ-GAN / VQ-VAE | ✅ | `src/models/vqvae.py`; GAN mode via `gan_weight` |
| 3D diffusion model | ✅ | `src/models/diffusion.py` (`DDPM`) |
| 3D U-Net | ✅ | `UNet3D` — compact single-downsample variant |
| **Spatial / depth attention** | ❌ **Not implemented** | `UNet3D` uses FiLM time/label conditioning only; no attention layers. Sole gap vs. the technique table |
| SSIM / PSNR | ✅ | `src/metrics.py` |
| MS-SSIM | ✅ (own implementation) | `evaluation.py:13-47` — adaptive 3D scales; window/level counts recorded in every report |
| Dice score | ⚠️ Helper only | `segmentation_overlap()` provided; MRNet supplies no masks, so the implemented downstream task is classification (documented) |

### 4.6 Compliance conclusion

The pipeline, methodology, and evaluation apparatus match the milestone document end-to-end.
For the Milestone 2 write-up, three things should be stated explicitly:

1. **Spatial/depth attention is not implemented** in the baseline U-Net — either add it or
   formally declare it out of scope for the compact baseline (currently the docs describe the
   U-Net accurately but never flag the omission against the reference).
2. **Preprocessing parity choice:** the default MRNet shape (128×128×32, resize) is a
   compute-driven deviation from the paper's center-crop 256×256×32; a paper-parity
   configuration is available by setting `--shape 256 256 32`.
3. **Results claims must remain scoped to smoke execution** until MRNet access and GPU
   training are resolved — the repository's own reports already enforce this.

---

## 5. Code quality assessment

### 5.1 Strengths

1. **Integrity/provenance discipline is exceptional** — sha256 on every artifact, re-verified
   at every stage boundary; traversal guard; atomic writes; safe deserialization everywhere.
2. **Reproducibility is engineered, not cosmetic** — seeded everything, RNG-state resume,
   determinism asserted in tests.
3. **Numerics are correct** — VQ straight-through estimator and chunked distance computation
   (`vqvae.py:26-38`), DDPM posterior mean/variance (`diffusion.py:58-84`), Mann-Whitney AUC
   (`downstream.py:36`), largest-remainder split allocation (`splitting.py:12-15`), float64
   for schedules and latent statistics.
4. **Scientific honesty in code** — smoke artifacts structurally excluded from research paths
   (`experiments.py:47,176`; `research_workspace.py:21`; `downstream.py:58-59`); every report
   carries explicit validity caveats; paired metrics are refused for unrelated generated/real
   pairs (`evaluation.py:80`).
5. **Robust data layer** — per-file error isolation, strict config validation, YAML-relative
   path resolution, subject-level leakage prevention enforced at dataset construction time.

### 5.2 Issue catalog

| # | Severity | Issue | Location |
|---|---|---|---|
| 1 | High (process) | **No version control** — nothing is committed despite a correct `.gitignore` | repository root |
| 2 | High (progress) | **CPU-only PyTorch** on a CUDA-capable machine blocks all real training | `requirements-lock.txt` (`torch==2.14.0`) |
| 3 | Medium | **No spatial/depth attention** in the U-Net — the one milestone-table technique absent | `src/models/diffusion.py:24-49` |
| 4 | Medium | **`evaluate_generation()` has zero test coverage** — the only untested pipeline stage | `src/training/evaluation.py:54-83` |
| 5 | Medium | **Dead duplicated finite-check** — re-checks the unchanged `loss` tensor after `optimizer.step()`; can never fire | `src/training/experiments.py:98-99` (live check at 91-92) |
| 6 | Medium | **No Streamlit caching** — `load_mri()` fully re-decodes the NIfTI on every rerun; the reconstruction expander re-hashes every processed tensor on disk per interaction | `app/streamlit_app.py:140`; `app/research_workspace.py:146` |
| 7 | Low | **Exact function duplication** — `source_hash()` ≡ `sha256_file()` | `src/data/pipeline.py:39-44` ≡ `src/utils/integrity.py:14-19` |
| 8 | Low | **Dead code + dependency split violation** — the `pytorch-msssim` wrapper is never called, yet the package sits in core `requirements.txt` | `src/metrics.py:21-35` |
| 9 | Low | **Naming collisions** — two unrelated `PLANES` constants (re-hardcoded a third time in the app); two parallel config validators with different shape rules; **two different definitions of "dataset_fingerprint"** (source-hash vs. tensor-bytes) | `src/data/mrnet.py:19`, `src/visualization/mri_viewer.py:5`, `app/research_workspace.py:124`; `src/utils/config.py:8-41` vs `src/training/common.py:82-92`; `src/data/pipeline.py:90` vs `src/training/common.py:46-57` |
| 10 | Low | **Silent error narrowing** — MRNet failures reduced to exception type name; corrupt run folders silently skipped in the app; MRNet page wraps its whole body in `except Exception` | `src/data/mrnet.py:83-84`; `app/research_workspace.py:25-26, 77-78` |
| 11 | Low | **Non-standard GAN update ordering** — discriminator steps before the generator's adversarial term is computed against it; choice is undocumented | `src/training/experiments.py:88-90` |
| 12 | Low | **TOCTOU in checksum cache** — stat-then-hash accepts a file modified between the two | `src/data/dataset.py:39-43` |
| 13 | Low | **Hardcoded thread count** ignores the config's `threads` key | `src/training/downstream.py:46` (contrast `experiments.py:25`) |
| 14 | Low | Minor unguarded edges: `half-1` division if `base ≤ 1`; modulo-wrap batch sampling can duplicate a sample when `batch_size > len(train)`; `report.json` read three times in one expander; one `AppTest.run()` call lacks the timeout used everywhere else; `smoke_research.py` crashes without `latest_run.json` | `diffusion.py:39`; `experiments.py:199`; `research_workspace.py:137,143,145`; `test_app.py:29`; `scripts/smoke_research.py:19` |
| 15 | Low | **Stale README** — test count says 39; the suite now contains 43 | `README.md` |

### 5.3 Test suite assessment

- **Strong:** data layer (validation codes, loader, 10 preprocessing tests incl. world
  coordinates and orientation), pipeline (split determinism, npy/pt parity, bad-file
  isolation), integrity (fingerprint mismatch, leakage rejection, escape guard), and a full
  synthetic end-to-end research chain with **resume** for both models and **seeded
  generation determinism** assertions.
- **Application testing is above average:** `AppTest` flows cover empty launch, both research
  pages, a full synthetic workflow including Export, and (when samples are present) a
  real-data page-by-page drive.
- **Gaps:** `evaluate_generation()` untested; GAN discriminator semantics and nonfinite-loss
  aborts unasserted; `downstream.MRIClassifier3D` only exercised at 2-step scale; CLI scripts
  and the downloader untested; the real-data tier is `skipif`-gated and never runs on clean
  clones; the "Model Experiments" reconstruction-pair viewer path (the most complex UI code)
  has no test with a verified run present.

---

## 6. Prioritized recommendations

| Priority | Action | Rationale |
|---|---|---|
| P0 | `git init` and make an initial commit | All work is currently unversioned; the `.gitignore` is already correct |
| P0 | Install a CUDA PyTorch build matching the RTX 5070 Ti | Unblocks the entire research program; `device: auto` will then select the GPU |
| P0 (external) | Complete the MRNet access application | Gates methodology step 1 and all real results |
| P1 | Decide on **spatial/depth attention**: implement, or document as a declared deviation | Aligns the code with the milestone technique table |
| P1 | Add tests for `evaluate_generation()` (manifest parsing, `max_samples` gating, diversity math) | Only untested pipeline stage |
| P1 | Add `st.cache_resource` around `load_mri` and cache fingerprint/checkpoint verification | First meaningful dashboard performance win; removes per-interaction full-volume hashing |
| P2 | Consolidate duplicates: one hash function, one `PLANES` map, one fingerprint concept (or rename the second) | Removes the top maintainer traps |
| P2 | Add a `configs/` paper-parity MRNet config (256×256×32) | Makes the methodology step-2 deviation opt-in rather than default |
| P3 | Remove the dead finite-check (`experiments.py:98-99`) and dead `ms_ssim`; move `pytorch-msssim` back to optional; fix README test count (39→43) | Hygiene |

---

## 7. Bottom line

The repository implements the Milestone 2 proposal faithfully and verifiably: the two-stage
latent-diffusion architecture, the paper's ×4 compression and codebook-snapped decoding, the
conditional 3D DDPM, and the full evaluation stack are all present, tested, and demonstrated
end-to-end at smoke scale with uncompromised provenance. The remaining distance to the
milestone's *research* outcomes is not code — it is the MRNet dataset, a CUDA-enabled
PyTorch install, and expert anatomical review. Within the code itself, the single technical
gap against the milestone document is the absence of spatial/depth attention in the U-Net,
which should either be implemented or formally scoped out; everything else in the issue
catalog is consolidation and hygiene.

*Analysis only — no repository files were modified in producing this audit.*
