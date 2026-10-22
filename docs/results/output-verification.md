# Output verification and practical usefulness

Verified on 22 October 2026 using the locally available **M4 96-training /
24-validation MRNet experiment**. The assessment checks existing data and models;
it does not introduce a new cohort or an independent-test claim.

**Conclusion:** the software and saved reconstruction scores are reproducible.
The VQ model is a useful representation-learning/reconstruction baseline, with a
real intensity-error improvement and a structural-detail tradeoff. The current
generator does not demonstrate useful anatomical synthesis or augmentation.

## 1. Software verification

| Check | Result |
|---|---|
| Full local suite, including available CUDA and live checkpoints | 144 passed; no skips |
| Automated statement/branch coverage | 72.60%, including the new output-verification script |
| Empty-data workspace with CUDA hidden | 138 passed, 6 expected skips |
| Empty-data CPU coverage | 70.52%; above the 70% gate |
| Mypy | Passed across 64 runtime source files |
| Ruff, installed dependency checks, diff whitespace checks | Passed |
| Built-wheel installation outside the checkout | Passed, including tiny CPU training/evaluation |

The new `scripts/verify_outputs.py` has independent numerical oracle tests. Its
real-data end-to-end execution is a separate audit, rather than being counted as
part of pytest's coverage percentage. The preceding 78.17% coverage result used
the smaller denominator before this auditor was added.

The empty-data run used a separate copy of the current source, no MRI/run folders,
and `CUDA_VISIBLE_DEVICES=''`. Its skips were local-data/hardware-dependent cases.
Both this run and the complete GPU run were Windows; hosted Linux execution is
defined by CI rather than represented as locally executed here.

## 2. Data, checkpoint and output integrity

The audit checked:

- All **360 source arrays**, their recorded hashes, source selection counts, and
  label support.
- All **120 processed tensors** and official exam split/label assignments.
  Independent raw-to-processed replay had **zero maximum absolute error**.
- All **120 cached latent files**, their hashes, membership in the selected
  codebook, and training-only normalization statistics.
- Both selected checkpoint hashes, dataset fingerprints, reconstruction/cache
  lineage, and validation-based checkpoint selection. VQ selected epoch 38/40;
  diffusion selected 283/300, with 1,920 and 14,400 completed optimizer updates.
- All **24 saved validation reconstructions** and all four per-exam/aggregate
  metrics. Maximum discrepancy against saved metrics was **1.32e-13**.
- All **8 saved generated tensors and latents**, including decoding their saved
  latents through the selected model. Maximum decoder replay MSE was **1.56e-9**.
- Another **27 saved samples**: 24 interactive, two remote seed-repeat samples,
  and the original seed-repeat sample. Tensor/latent hashes, ranges, shapes, and
  selected-model-pair references matched.
- The 24 remote-validation reconstructions: all 96 metric values reproduced to
  within **1.36e-13**. The folder name does not make these an independent cohort;
  they reference the same validation inputs.

### Reproducibility scope

Fresh generation for seeds 2026–2033 was repeated twice on the current runtime.
All eight image/latent pairs were **byte-identical between the two repetitions**;
the eight different seeds produced distinct latent files.

Historical artifacts used an RTX 5070 Ti/PyTorch 2.11 environment; this verification
used an RTX 3060 Laptop/PyTorch 2.14.1 environment. Cross-environment replay is not
byte-identical: mean latent code agreement was **99.962%**, with maximum latent
MSE **2.50e-5**. A CPU and TF32-disabled probe did not eliminate that assignment
drift. Current validation PSNR/SSIM were **24.29983 / 0.76588**, compared with the
saved **24.29968 / 0.76587**. These observations support stable aggregate results,
not a cross-hardware bitwise-reproducibility claim.

Initial diagnostic attempts stopped at a strict cross-environment replay tolerance
of 1e-6 MSE. The completed audit records historical replay differences explicitly;
strict file-integrity and same-runtime repeatability checks remain separate.

## 3. Is reconstruction useful?

Independent recomputation on the same 24 validation exams:

| Metric | VQ reconstruction | Half-resolution interpolation | Train-mean image |
|---|---:|---:|---:|
| MSE ↓ | 0.015368 | 0.029747 | 0.090409 |
| MAE ↓ | 0.074935 | 0.089197 | 0.209265 |
| PSNR ↑ | 24.29968 dB | 21.36067 dB | 16.49929 dB |
| SSIM ↑ | 0.76587 | 0.77707 | 0.30274 |

VQ reduces mean MSE by **48.34%** and improves MSE/MAE/PSNR on **23/24** exams.
However, it improves SSIM on only **4/24** exams. Paired exam-bootstrap intervals
(10,000 resamples) for VQ minus interpolation are:

- PSNR: **+2.939 dB**, 95% interval **[+2.487, +3.341]**.
- SSIM: **−0.01120**, 95% interval **[−0.01628, −0.00635]**.

This is a consistent intensity-error benefit with a consistent SSIM penalty on
this cohort. Contact-sheet inspection agrees: reconstructions retain broad knee
structure but smooth fine boundaries and textures. They do not uniformly improve
image quality. Interpolation and learned codebook representations have different
information/parameter budgets; this is not a rate-distortion codec comparison.

Training reconstruction MSE was 0.015383, close to validation MSE. Tokenizer usage
was 125/128 codes, aggregate perplexity 70.27, and dead-code fraction 2.34%.
These are useful engineering observations, not proof of pathology preservation.
Patient linkage is absent; intervals are exam-level and validation selected the
checkpoint, so they do not establish independent generalization.

## 4. Is generation useful?

Saved and fresh generated contact sheets were inspected at fixed intensity scale
and slice fractions. Samples remain blurred, texture-like, and lack reliable
coherent gross joint structure in the inspected panels. This was engineering
visual inspection, not expert anatomical scoring.

- Adaptive pairwise MS-SSIM was **0.22963** for generated samples versus **0.32260**
  for eight class-matched real training references. The real-reference selection
  differs from the historical report's selection, so its old 0.30247 value is not
  treated as the same comparison.
- Mean generated gradient energy was **16.1%** of the real-training mean,
  consistent with substantial smoothing. Neither this descriptor nor low
  pairwise MS-SSIM establishes realism.
- Nearest-neighbor search covered all **96 generator-training exams**. Nearest
  raw-pixel MSE ranged from **0.11932 to 0.14728**. No exact copies were found by
  that screen; this does not establish anatomical quality or exclude all memorization.
- Changing the requested condition from `[1,0,0]` to `[1,1,0]` with identical noise
  changes the output, but does not visibly/quantitatively establish an ACL lesion.
  The alternate joint condition has only seven supporting training exams.

Thus reproducible sampling and diversity diagnostics are functioning, while useful
anatomical synthesis and condition fidelity are not established by these outputs.

## 5. Does synthetic augmentation help?

Nine small classifiers were fitted: **three seeds (42, 43, 44), three arms, 400
optimizer updates per arm**. The comparison uses the existing 96/24 cohort and
saved eight synthetic volumes. A real-data control matches the synthetic joint
label distribution on the same odd updates. Initializations and update budgets
are matched; real draws on even updates match the augmented arm.

Validation predictions were reconstructed from saved classifier weights and all
metrics independently checked using pairwise positive/negative comparisons for AUC.

| Mean validation AUC across seeds | Real only | Real + synthetic | Label-matched real resampling | Constant-prevalence baseline |
|---|---:|---:|---:|---:|
| Abnormal | 0.7614 | 0.5860 | 0.7684 | 0.5000 |
| ACL | 0.2517 | 0.4219 | 0.2704 | 0.5000 |
| Meniscus | 0.2786 | 0.3405 | 0.2857 | 0.5000 |

Important interpretation:

- At the declared threshold 0.5, **all nine models predicted every validation
  case as abnormal and every case as ACL-negative/meniscus-negative**. Accuracies
  of 79.17%, 54.17%, and 58.33% merely match majority prediction; ACL and meniscus
  sensitivity are zero.
- The ACL AUC increase has a nominal paired-exam interval excluding zero
  (**+0.1702**, interval **[+0.0115, +0.3637]** versus real-only). Its absolute AUC
  remains below 0.5, and calibration worsens. It is not evidence of a useful classifier.
- Mean Brier error worsens from **0.3409 to 0.3663** for ACL and from **0.2577 to
  0.2945** for meniscus. Abnormal AUC worsens substantially.
- All eight synthetic labels request `[1,0,0]`. Mixing them into half the updates
  changes training ACL-positive frequency from **18.75% to 9.375%** and meniscus
  frequency from **35.42% to 17.71%**. These generated labels are requested conditions.
- The small classifier/budget is weak. A separate **training-only positive
  control** selected one real case per observed joint-label stratum (five cases).
  At 1,000 cyclic updates, all three tasks reached training AUC/accuracy 1.0 and
  cycle BCE fell from 0.7118 to 0.0291. This confirms a working training path;
  memorizing five cases does not establish generalization on 96/24 exams.

The supported conclusion is **no demonstrated practically useful augmentation in
this experiment**. The limited cohort, weak probe, and unverified condition fidelity
also prevent a general claim that all synthetic augmentation would fail.

## 6. Useful today and evidence boundaries

| Capability | Assessment |
|---|---|
| MRI validation, preprocessing, visualization and provenance | Useful local research tooling; checked by fixtures and real-data replay |
| VQ representation learning and reconstruction | Useful research/teaching baseline with measured intensity/structure tradeoffs |
| Reproducible diffusion experiments | Useful for investigating sampler/model failures |
| Realistic synthetic knee MRI / verified disease conditioning | Not demonstrated by the available outputs |
| Beneficial classification augmentation | Not demonstrated; current probe has no useful positive operating point at threshold 0.5 |
| Published research-v2 256/60, three-seed result | Local data/checkpoints/study artifacts were unavailable; not independently reverified here |

## Reproduction and local evidence

```bash
python -m pytest -q --cov --cov-report=term-missing
python -m mypy
python -m ruff check .
python -m pip check
python scripts/verify_outputs.py --run runs/M4-MRNET-20261021 --output runs/OUTPUT-VERIFICATION-new --device cuda --steps 400 --seeds 42 43 44
```

The completed local audit is `runs/OUTPUT-VERIFICATION-20261022-03/`:
`plan.json`, `audit.json`, per-exam reconstruction results, refreshed generation
diagnostics, tokenizer counts, classifier weights/predictions, and saved/fresh
contact sheets. `additional-checks.json` records auxiliary output verification and
the five-case positive control. Runtime manifests, arrays, images, weights and
local paths remain in ignored artifact directories; this document publishes aggregates.
