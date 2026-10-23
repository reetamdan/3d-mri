# Knee representation and latent-diffusion research study

## Outcome

The compact **anisotropic KL representation improves internal reconstruction**:
across three training seeds it reaches **26.802 ± 0.199 dB PSNR** and
**0.8578 ± 0.0092 SSIM**, versus **24.352 ± 0.100 dB / 0.7691 ± 0.0081** for the
matched native VQ control. All 15 completed representation runs used 1,920
optimizer updates and effective batch two. Their checkpoints, actual update
counts, per-exam intervals and metrics are recorded in the generated
[measured table](results/ae_bakeoff_measured.md) and its JSON companion.

**No representation passes the predeclared fine-detail gate.** The strongest
single-seed reconstruction reaches 29.072 dB / .9158 SSIM only by reducing lattice
compression to 2×2×2, with no float-scalar compression at eight latent channels;
its gradient/HF ratios still fail. These results justify a better reconstruction
candidate, not a claim of preserved ACL/meniscus pathology or useful synthetic MRI.

## Starting question and preserved baseline

Can a compact continuous KL representation preserve more structure than the
existing VQ bottleneck on a 6 GB RTX 3060 Laptop-class device? The exact starting
map, checkpoint hashes, configurations and 96/24 cohort fingerprint are recorded
in `docs/current_baseline.md`. Engineering work was pushed to main at `b824aef`,
with clean-build/clean-checkout fixes `680033b` and `ed7a13e`; the research branch
starts from `ed7a13e`. Existing checkpoint/data outputs remain separate from new
study outputs. External source/license/paper review is in
`docs/external_repos_review.md`.

## Data audit (completed before candidate validation)

`scripts/audit_knee_data.py` verified all 120 processed tensors. Raw acquired
slice counts range **17–49**, resized to **128×128×32**. Tensor order is
`[B,C,row,column,acquired-series slice]`. Preprocessing transposes the source
`[slice,row,column]`, trilinearly resizes with `align_corners=False`, then applies
per-volume min-max normalization to [-1,1]. There is no explicit crop/pad or
programmed left/right flip. Resizing itself loses information.

Physical orientation, mm spacing, laterality, reliable scanner/sequence metadata
and patient linkage are unavailable in these MRNet arrays. The audit proves the
array-grid contract, not patient-space orientation consistency. Spacing plots
and anatomical ROI labels are therefore unavailable, not fabricated. Source and
processed slice comparisons, central tensor-axis views, slice-count/intensity
histograms and all per-exam flags are in `runs/REPRESENTATION-DATA-AUDIT/`.

## Predeclared autoencoder design

Primary screening: seed 42, same frozen 96 training / 24 validation exams,
**40 epochs × 48 updates = 1,920 updates per arm**, effective batch two, Adam
3e-4, gradient clipping 1, full 128×128×32 inputs. Every arm selects by the same
validation L1 rule. Historical checkpoint replay is reported separately from
the newly trained matched-budget native VQ control.

| Arm | Controlled question |
|---|---|
| `ae_vq_baseline` | Native 128-code VQ architecture/objective, matched reproduction |
| `ae_vq_improved` | Change only codebook capacity 128→256; no simultaneous EMA/GAN/decoder changes |
| `ae_kl_iso` | Official MONAI compact KL, 4×4×4 lattice compression |
| `ae_kl_aniso` | Same parameters and initialization, only 4×4×2 sampling/upsampling |
| `ae_kl_aniso_ssim` | Add adaptive 3D MS-SSIM objective |
| `ae_kl_aniso_structural` | Add three-axis finite-difference preservation to the preceding objective |

KL uses MONAI 1.6.1, channels 8/16/32, eight latent channels, GroupNorm four groups,
one residual block per level, no attention/nonlocal blocks, a tanh output to
maintain the existing [-1,1] contract, sampled training and posterior-mean
reconstruction. Only sampling pad/stride/upsample scales are adapted for
anisotropy. Tests confirm identical initial parameters for the two lattice arms.

Native VQ retains full precision; KL uses mixed precision for feasible memory and
runtime. This difference is disclosed, not a claim of matched floating-point
arithmetic/FLOPs. Parameters are **214,697 / 215,721** for the two VQ arms and
**196,409** for the primary KL arms. Scalar latent counts reduce inputs 8× for isotropic
and 4× for anisotropic KL; continuous payload and discrete-code bitrates are not
equated. No practical compression-codec result is inferred.

### Loss decisions frozen from training data

KL regularization uses **mean** over batch/channel/spatial entries with coefficient
1e-6. MONAI's tutorial uses a spatial **sum**, also with 1e-6; those numbers are not
interchangeable. Our dimension-invariant weak regularizer avoids doubling its
strength when the slice lattice doubles. Actual posterior statistics and latent
scaling must be measured; unit-variance Gaussian latents are not assumed.

The structural coefficients are computed once from the first training batch's
initial deterministic reconstruction, balancing output-gradient norms: MS-SSIM
gets 25% of the L1 output-gradient norm; the gradient term gets either 0% or 25%.
Coefficients/norms are persisted. No validation images or candidate validation
scores choose these weights. MS-SSIM uses Gaussian window 3 with adaptively
truncated/renormalized standard weights because 32-slice inputs cannot support
standard five-scale 11-voxel windows. Numerical tests compare the implementation
with the existing reference-checked metric. Finite differences use a common unit
tensor grid, not invented physical spacing. No perceptual/GAN term is active in
this first comparison. Optional perceptual hooks require explicit frozen weights
and their checksum.

### Go/no-go gate (frozen before candidate validation)

The full gate is in `runs/AE-BAKEOFF/plan.json` and source `GATE`:

- Mean PSNR at least frozen-VQ PSNR minus 0.1 dB.
- Mean SSIM and adaptive MS-SSIM at least the same-cohort interpolation reference.
- Mean squared-gradient energy ratio in [0.6,1.3] and at least 20% closer to one
  than frozen VQ; unit-grid edge MAE no worse than frozen VQ.
- High-frequency energy ratio in [0.5,1.5], using radial array-grid frequency ≥0.35.
- Inspect fixed frozen-baseline worst/median/best and seed-42 random cases, full
  slices, fixed central grid crops and absolute errors; no severe new artifacts.
- Anatomy/lesion masks and validated knee-model proxy scores are unavailable.
  Passing this engineering gate cannot be reported as verified ACL/meniscus preservation.

If the quantitative/visual gate fails, major new diffusion/downstream training is
not justified. The thresholds will not be moved after seeing candidate scores.

## Compute smoke results (training exams only)

One-batch → 10-update → 100-update finite forward/backward checks completed for
all six arms on the RTX 3060. These are repeated training inputs, not quality results.

| Arm | Latent [C,row,column,slice] | Peak allocated MiB | Median seconds/update | Estimated training-only seconds at 1,920 updates |
|---|---|---:|---:|---:|
| VQ baseline | 8×32×32×8 | 202.9 | 0.0346 | 66 |
| VQ 256 codes | 8×32×32×8 | 202.9 | 0.0347 | 67 |
| KL isotropic | 8×32×32×8 | 750.5 | 0.1037 | 199 |
| KL anisotropic | 8×32×32×16 | 774.5 | 0.1063 | 204 |
| KL anisotropic + MS-SSIM | 8×32×32×16 | 814.9 | 0.1546 | 297 |
| KL anisotropic + MS-SSIM + gradient | 8×32×32×16 | 816.9 | 0.1578 | 303 |

Validation, metrics, checkpoint I/O and full-cohort input loading add runtime.
The six-arm primary study fits available memory without patches/checkpointing;
estimated checkpoint/reconstruction storage is below 1 GB. No high-resource
architecture is needed to test the first representation hypothesis.

## Completed representation experiments

Training and model selection use the same 96/24 cohort throughout. The 24 exams
are **model-selection validation**, not independent patients or a final test.
Three-seed values below are actual mean ± sample SD; the other arms have one seed.

| Representation | Seeds | PSNR dB | SSIM | Gradient ratio | HF ratio | Quantitative gate |
|---|---|---:|---:|---:|---:|---|
| Native VQ, 128 codes | 42/43/44 | 24.352 ± .100 | .7691 ± .0081 | .314 | .108 | 0/3 |
| KL isotropic, 4×4×4 | 42/43/44 | 25.502 ± .070 | .8169 ± .0037 | .358 | .097 | 0/3 |
| KL anisotropic, 4×4×2 | 42/43/44 | **26.802 ± .199** | **.8578 ± .0092** | .467 | .160 | 0/3 |
| VQ, 256 codes | 42 | 24.640 | .7864 | .335 | .111 | Failed |
| Anisotropic KL + MS-SSIM | 42 | 27.003 | .8734 | .501 | .145 | Failed |
| Anisotropic KL + MS-SSIM + gradient | 42 | 26.981 | .8745 | .510 | .157 | Failed |
| Preceding structural arm + MedicalNet loss | 42 | 26.891 | .8625 | .485 | .149 | Failed |
| KL low-compression, 2×2×2 | 42 | 29.072 | .9158 | .570 | .255 | Failed |
| Anisotropic KL, learned deconv upsampling | 42 | 26.463 | .8463 | .459 | .134 | Failed |

The anisotropic arm improves mean PSNR by **2.451 dB** and SSIM by **.0887** over
the matched native control across seeds. The seed-42 anisotropic model versus
the **historical** VQ checkpoint improves paired PSNR by 2.729 dB, with a
descriptive 95% exam-bootstrap interval [2.541,2.921], and SSIM by .1023
[.0962,.1086]. Those intervals use 24 resized exams, 2,000 resamples and a fixed
seed; they neither account for unknown patient clustering nor remove model
selection bias. Seed variance and exam-bootstrap uncertainty are different.

The perceptual, lower-compression and learned-decoder arms are explicitly
single-seed **post-screen exploratory contingencies**, not additions retroactively
claimed as part of the original six-arm plan. Each has its own persisted plan.

The structural losses improve single-seed SSIM over plain anisotropic KL, but
not PSNR or HF preservation. Adding verified frozen medical features does not
improve this arm. Learned upsampling reduces parameters to 167,993 but worsens
reconstruction. The 2×2×2 experiment demonstrates a representation/detail
tradeoff, not an efficient latent codec: its `[8,64,64,16]` tensor contains as many
float scalars as the `[1,128,128,32]` input. All variants remain below gradient
ratio .6 and HF ratio .5. No threshold was relaxed after seeing these results.
These engineering detail proxies also respond to source-image noise; they are
not clinical or lesion-specific measures, and their failure alone cannot identify
which tissue or pathology is lost.

### Visual evidence and unavailable anatomy labels

Each study saves the same frozen-VQ worst/median/best and seed-42 random cases in
`comparison-*.png`, `errors-*.png`, `grid-roi-*.png` and `visual-cases.json`.
The selection contains five unique exams because a random case can coincide
with a ranked case. Array-axis views show better contour/detail retention for
anisotropic/low-compression KL, with residual smoothing; the learned-upsample
contingency is visibly softer. These are informal computational observations.
The central grid crop is not an annotated ACL/meniscus ROI, and no expert
diagnostic review has been completed. Study records retain visual review as
pending; none receives a passed representation receipt.

### Actual resource use and failed execution attempts

The measured research device is an **RTX 3060 Laptop, 6 GiB VRAM**, with roughly
16 GiB host RAM, torch 2.14.1+cu130 and MONAI 1.6.1. Peak *allocated training*
memory is about 203 MiB for VQ, 752 MiB KL iso, 776 MiB KL aniso, 809 MiB
structural, 1,053 MiB perceptual, 918 MiB low-compression and 741 MiB deconv.
These allocator measurements exclude driver/context overhead and host storage.

Concurrent training/verification exhausted host committed memory despite spare
VRAM. One latent-cache diagnostic attempt stopped with `MemoryError` and is
retained under `runs/DIFFUSION-SANITY/`; it is excluded from completed results.
Subsequent heavy jobs run serially with lazy CUDA module loading, cuDNN plan
cache capped at 128 and oneDNN primitive cache capped at 32. Each run records its
environment, source-file hashes, configuration and checkpoint lineage. Patches
and activation checkpointing were not needed for the completed full-volume AE
comparison.

## Frozen representations, scaling and medical-feature diagnostics

`cache-representation` now handles posterior-mean KL and quantized VQ. It fits
per-channel mean/std and bounds on **training latents only**, checks tensor
geometry/cohort/model hashes, and carries external pretraining exposure forward.
Legacy scalar normalization remains supported unchanged. VQ caches can report
dead codes, entropy/perplexity and requested-label-stratified code counts.

For the seed-42 anisotropic KL used in tiny-data diagnostics, actual channel
means span **−2.301 to 3.105**, standard deviations **.621 to 2.286**, and mean
posterior sigma is **.02070**. Global mean/std are −.2253 / 2.2341. The weak KL
regularizer clearly does not make these posterior means independent standard
normal latents; measured normalization is necessary. Pooled per-label means are
descriptive summaries, not evidence of pathology encoding.

The official public MedicalNet ResNet10 checkpoint was explicitly downloaded at
revision `deeeaa1e2a342b63e28b61e9d993b63e79c4b437`; its weight SHA-256 is
`afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605`.
The exported frozen descriptor has no classification head, permutes to
depth/row/column, resizes to 32×64×64 and volume z-scores. Its local export hash is
`8e319528e3a8265beef3cf25b145a5afb040a1369b99c3ee2474ec33a18788fd`.
Exports may have different archive hashes when regenerated; always bind the
actual export hash in its own provenance/configuration.

`runs/AE-FEATURE-AUDIT/` evaluates the six primary seed-42 reconstructions and
eight **historical** generated volumes against all 96 real training exams using
frozen medical-feature neighbors, exhaustive full-volume SSIM neighbors and
eight octants per volume with 8×8×4 pooled-pixel patch neighbors. The paired
feature cosine distance is 2.23e−5 for native VQ and 7.71e−6 for anisotropic KL;
these very small, extractor-dependent values are not a knee-validated metric.
None of the eight historical generations is an exact file duplicate or passes
the descriptive proximity flag. The flag threshold is the first percentile of
same-cohort validation-to-training feature distances, not a privacy guarantee.
Class-matched distribution estimates use only 37 real / eight generated vectors
in 512 dimensions and are covariance-rank limited; they are not a stable FID
benchmark. Feature descriptors, neighbors, hashes and contact sheets are retained.

Actual tissue Dice/IoU/volume/boundary or lesion-preservation scores remain
unavailable. `src/training/anatomy.py` accepts supplied aligned masks, but this
adapter and its artificial numerical tests do not supply missing clinical evidence.

## Diffusion: implemented components and tiny-data evidence

Published Min-SNR equations are locally implemented and target-specific tests
cover epsilon, v and x0 weights. The denoiser's supported prediction modes remain
epsilon/v. We weight each volume's MSE before batch averaging and retain raw
unweighted denoising error for comparisons. Optional bottleneck condition-token
cross-attention, null-token dropout/CFG, KL decode and train-fitted per-channel
scaling are implemented with small integration tests. Their full-cohort presets
are **configurations, not trained results**.

The deliberately artificial ±.5 conditional-constant control learns both targets
in 500 updates and passes its fixed mean/MSE checks. This validates a numerical
conditioning path; it is not an MRI/pathology experiment.

The completed `runs/DIFFUSION-SANITY-02/` diagnostic intentionally fits **one
training MRI posterior mean**. It uses a base-16 `[1,2,4]` UNet, v-prediction,
32-step cosine schedule, EMA .995, mixed precision, Adam 3e−4, 2,000 updates,
batch two, DDIM32 and four fixed generation seeds. Normalization is frozen from
the 96 training latents. Samples are paired to the intentionally memorized
training reconstruction, not to held-out anatomy.

| One-training-volume diagnostic | Mean SSIM to its reconstruction | Normalized latent MSE | Fixed-noise unweighted denoising MSE | Gate |
|---|---:|---:|---:|---|
| Unweighted v baseline | .63788 | .32813 | .34110 | Failed |
| Min-SNR gamma 5, identical budget | .53915 | .41431 | .49118 | Failed |
| Array-coordinate v baseline | .66138 | .30715 | .33367 | Failed |
| Array coordinates + Min-SNR gamma 5 | .57944 | .37355 | .45759 | Failed |

The fixed technical gate requires SSIM ≥.9 and latent MSE ≤.1; four-volume fitting
is attempted only if one-volume passes, and must cover at least two cases.
All four arms failed, so the four-volume runs were skipped. Min-SNR was worse in this
configuration. This is a bounded negative result, not a general claim about the
method. Full-cohort training remains unauthorized by the representation gate.

The completed `runs/DIFFUSION-COORD/` diagnostic tests a spatial-inductive-bias
hypothesis using three normalized **array-index** channels, zero-initialized extra
input columns and exactly matched shared initial parameters/RNG. It reuses the
same integrity-checked latent cache and does not infer anatomical coordinates.
It adds 1,344 parameters (857,832 total) and modestly improves the unweighted
one-volume score, but does not pass. The four fixed samples still have substantial
texture/structure mismatch to their training reconstruction. Both ordinary and
coordinate experiments pass the artificial conditional-constant control; that
contrast demonstrates why a trivial conditioning fixture cannot establish MRI
memorization or MRI generation quality. No larger diffusion sweep followed.

## Optional external pretraining and fair downstream framework

The local fastMRI/SKM-TEA HDF5 adapter requires explicit acquisition, image key,
volume split and a patient ID where available. It accepts magnitude targets or
documented image echoes, rejects raw complex k-space, keeps volumes intact,
checks patient grouping, records source hashes and never assigns MRNet pathology
labels. SKM-TEA six-channel tissue masks use aligned nearest-neighbor resampling.
`init_checkpoint` requires the same autoencoder architecture/grid, records all
pretraining fitting/selection exposure and rejects overlap with MRNet validation.
**No actual external representation pretraining was performed:** authorized
compatible source images were not available locally.

`scripts/downstream_study.py` implements five-seed defaults for `real_only`,
stable-TorchIO and label-matched real-resampling controls, with gated synthetic
and optional combined arms. All share classifier initialization, real draw
indices, optimizer-update/effective-batch budgets. Odd replacement updates use
the same uniformly selected observed joint-label strata for resampling/synthetic
arms. TorchIO uses conservative in-plane unit-grid affine and weak bias/noise
on positive intensities; no flips or physical-mm interpretation are assumed.

Thresholds are fixed by development-only Youden index, with documented tie
handling. Reports include AUROC/AUPRC, sensitivity, specificity, balanced
accuracy, Brier score, bootstrap intervals and actual seed SD. Separate final
evaluation checks identity/content exposure; unknown patient linkage prevents a
patient-independent claim. Without a separate cohort, thresholded scores are
explicitly labeled development-only. Synthetic requested labels are not expert
pathology labels, and pools must cover every observed joint stratum and present
a passed generation-quality receipt bound to the models.

These contracts have small artificial integration/regression tests; **no serious
new downstream synthetic-utility experiment was run**. The earlier M4 negative
augmentation probe is preserved in `docs/results/output-verification.md`; its
results do not substitute for the stronger conventional baseline implemented here.

## Decision and prioritized next research

1. Retain the frozen VQ benchmark and compact anisotropic KL as measured internal
   reconstruction candidates. The latter improves multiple seeds at low memory,
   but has not earned a generation/anatomy gate pass.
2. On this hardware, target the remaining detail loss with a prespecified
   acquisition-compatible pretrained representation or bounded loss/decoder
   hypothesis, retaining a from-scratch matched control. The completed low-
   compression/deconv/perceptual contingencies already limit optimistic claims;
   do not turn another validation sweep into an independent test.
3. The bounded spatial-memorization diagnostic is complete and failed; stop major
   diffusion sweeps until the representation and tiny-data gates pass. Full-cohort Min-SNR,
   epsilon/v, CFG strengths and cross-attention require distinct matched runs.
4. On an ideal larger-resource path, obtain compatible authorized knee volumes,
   reliable patient/acquisition metadata, aligned masks/validated knee encoders,
   a reserved independent cohort and blinded expert review. Evaluate full anatomy,
   local small structures, multimodal labels and memorization before claiming
   synthetic augmentation utility. Dense high-resolution or patch-volume models
   require their own measured compute/context-boundary comparisons.

No independent generalization, ACL/meniscus preservation, medically valid
generation, or positive synthetic augmentation claim follows from this study.

## Completion and verification scope

| Status | Work |
|---|---|
| Implemented and measured | Data audit; six-arm matched screen; three-seed VQ/iso/aniso confirmation; three one-seed contingencies; train-only frozen-cache statistics; explicitly pinned MedicalNet descriptor/perceptual comparison; all-training feature/SSIM/patch-neighbor audit of historical samples; four failed 2,000-update MRI-latent diagnostics and artificial conditioning controls |
| Implemented with small software tests, not a completed research result | Authorized-local external pretraining adapter/initialization; supplied-mask anatomy proxies; full-cohort Min-SNR/v/CFG/cross-attention configurations; balanced observed-label generation; stronger five-seed conventional/resampling/synthetic downstream framework |
| Researched and deferred | Larger external pretraining, annotated knee-specific validation, patch-volume/high-resolution/global-local alternatives, residual VQ/FSQ, unavailable TCAM-Diff source, EDM-compatible training, expert review and independent clinical testing |

On 24 October 2026, the full local Windows/CUDA suite passed **184 tests** with
**72.88% automated statement/branch coverage** over `src/`, `app/`, `mri3d/`
and `scripts/`. This includes isolated wheel install/preset resolution, existing
dashboard/live-model acceptance and the new scientific contracts. The tests
include artificial fixtures to validate software behavior; their numerical outputs
never enter the research result tables. Real training/audit executions above are
separate evidence, not counted as automated-suite coverage. The required floor
remains 70%; the current source passes mypy across 81 files and Ruff.
A fresh detached checkout without local data/models and with CUDA hidden passed
**178 tests, six expected skips and 71.42% coverage**. Dependency consistency and
a separate PEP 517 isolated wheel build also passed. Full details and exact scope
are in `docs/engineering-verification.md`.

## Reproduce the frozen screening plan

These commands assume the processed 96/24 cohort and historical VQ checkpoint
described in `docs/current_baseline.md` are available locally. Data and trained
weights are local artifacts. Use the baseline workflow in the README to prepare
them, or pass explicit `--data`/`--baseline` paths for a newly recorded study.

```bash
python -m pip install -e ".[dev,metrics,research]"
python -m pip install -r requirements-research.txt
python scripts/audit_knee_data.py --output runs/REPRESENTATION-DATA-AUDIT-new
python scripts/representation_study.py --output runs/AE-BAKEOFF-new --profile-only --profile-updates 100
python scripts/representation_study.py --output runs/AE-BAKEOFF-new --resume
python scripts/representation_study.py --output runs/AE-CONFIRMATION-new --arms ae_vq_baseline ae_kl_iso ae_kl_aniso --seeds 43 44
python scripts/export_medical_features.py --output runs/MEDICALNET-FEATURES-new
python scripts/representation_study.py --output runs/AE-PERCEPTUAL-new --arms ae_kl_aniso_perceptual --feature-extractor runs/MEDICALNET-FEATURES-new/features.pt
python scripts/representation_study.py --output runs/AE-LOW-COMPRESSION-new --arms ae_kl_low_compression
python scripts/representation_study.py --output runs/AE-DECODER-new --arms ae_kl_aniso_deconv
python scripts/summarize_representation.py --studies runs/AE-BAKEOFF-new runs/AE-CONFIRMATION-new runs/AE-PERCEPTUAL-new runs/AE-LOW-COMPRESSION-new runs/AE-DECODER-new --output docs/results/ae_bakeoff_reproduced.md
```

The installed `research` extra selects compatible ranges;
`requirements-research.txt` records direct package pins used in the local study.
Install a suitable PyTorch CPU/CUDA build first. Run heavy commands serially.
On constrained Windows/CUDA hosts, the measured memory policy can be set with:

```powershell
$env:CUDA_MODULE_LOADING='LAZY'
$env:CUDNN_V8_API_LRU_CACHE_LIMIT='128'
$env:ONEDNN_PRIMITIVE_CACHE_CAPACITY='32'
```

After a completed primary study and explicit descriptor export:

```bash
python scripts/evaluate_representation_features.py --study runs/AE-BAKEOFF-new --extractor runs/MEDICALNET-FEATURES-new/features.pt --output runs/AE-FEATURE-AUDIT-new --device cuda
python scripts/diffusion_sanity.py --autoencoder runs/AE-BAKEOFF-new/ae_kl_aniso-42/training/best.pt --output runs/DIFFUSION-SANITY-new --updates 2000 --device cuda
python scripts/diffusion_sanity.py --autoencoder runs/AE-BAKEOFF-new/ae_kl_aniso-42/training/best.pt --latent-cache runs/DIFFUSION-SANITY-new/full-cache --output runs/DIFFUSION-COORD-new --updates 2000 --device cuda --coordinates
```

The feature command defaults to the historical M4 generated outputs; supply
`--generated` explicitly for a different verified manifest. Diagnostic runs are
explicit smoke/training-reproduction artifacts and cannot authorize a full-cohort
gate. New output directories are immutable; use explicit `--resume` only for an
unchanged representation-study plan. Exact scores can drift across hardware,
runtime versions and source changes; archive hashes are not portability guarantees.
