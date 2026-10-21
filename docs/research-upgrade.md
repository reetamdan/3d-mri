# Research v2: implemented changes, evidence, and remaining requirements

This revision addresses the research review without treating software features as
substitutes for independent data or qualified anatomical review. The M4 baseline
and its recorded results remain reproducible historical experiments.

## What changed

| Review weakness | Implemented response | Evidence / remaining boundary |
|---|---|---|
| Evaluation required the training fingerprint | Separate training/evaluation identities; audit training **and model-selection** cases; optional frozen test protocol and exclusive evaluation receipt | External-cohort and leakage regression tests. A genuine new patient-disjoint cohort is still required |
| Only 96/24 exams and one training run | Seeded 256/60 sagittal cohort; matched-budget baseline/upgraded runs for seeds 42, 43, 44 | Aggregate results are generated from the completed local study; not the full MRNet cohort |
| Tiny single-scale denoiser | Optional multiresolution U-Net with widths 32/64/128 and bottleneck attention | Original architecture/checkpoints remain supported |
| No EMA or training stabilization | Warm-started denoiser EMA, optional EMA codebook/reset, mixed precision, gradient accumulation, plateau scheduling, early stopping | Exact CPU resume-equivalence and CUDA forward/backward tests |
| No conditional-guidance or fast-sampler experiments | Learned null conditioning with label dropout, CFG, epsilon/v targets, DDPM/DDIM with explicit training/inference steps | Algebraic denoising oracle tests; fixed-checkpoint inference-ablation runner |
| Codebook health unmeasured | Whole-cohort usage counts, dead-code fraction, aggregate perplexity, quantization error | `mri3d tokenizer-diagnostics`; report available in model workspace |
| Memorization screen searched only a few real exams | Exhaustive search across the recorded generator training cohort, independent of diversity cap | Exact-copy regression beyond the cap; raw-pixel and pooled-spatial nearest neighbours |
| Similarity numbers lacked uncertainty | Exam/patient-cluster bootstrap, per-exam and foreground reconstruction diagnostics; matched-seed study summaries | Patient identities absent in MRNet: exam intervals do not account for unidentified repeat patients |
| Custom MS-SSIM lacked reference tests | Single-scale and five-scale comparisons to `pytorch-msssim` under matching windows/scales/weights | Protocol still differs from papers with other windows or scales |
| No feature-distribution evaluation | Fréchet feature distance and unbiased polynomial KID on caller-supplied embeddings, with hashes and extractor ID | No medical feature extractor is automatically downloaded or declared valid; no medical FID result is invented |
| Conditioning not verified pathology | Recorded joint-condition support; same-noise condition-change experiments; blinded-review packet tools | Generated disease fidelity still needs independent assessment |
| Downstream results only point estimates | Sensitivity/specificity at declared threshold, bootstrap intervals, optional separate evaluation cohort | Scores describe actual completed experiments only; no augmentation benefit is assumed |
| Latent I/O overhead | Integrity-checked contiguous per-split NPY memory maps, retaining per-scan files/legacy fallback | Packed-cache equivalence and corruption tests |
| Training checkpoints unnecessarily large for inference | Compact model-pair export removes optimizer/history/RNG state and records original hashes | Exported samples are regression-tested against the source EMA pair; compact exports cannot resume training |
| Moving a run broke paths | Owner-relative references and `mri3d relocate`; checkpoint hashes preserved | Regression tests move artifact trees and resolve legacy Windows references |
| No installable project | `pyproject.toml`, editable install, `mri3d`, `python -m mri3d`, `mri3d-study` | CLI verified outside the repository directory |
| CI covered only offline execution | Ruff, focused mypy, branch coverage gate, Linux/Windows CPU matrix; manual trusted-main GPU workflow | Manual GPU workflow requires an operator-provided runner and is not claimed as hosted GPU CI |
| Documentation drift | Updated MRNet integration status and research guide; historical results labeled separately | Official Redivis delivery still needs separate archive-format integration |

## Install / upgrade after pulling

From the project directory, activate the existing virtual environment and run:

On Windows, stop this project's running Streamlit process before upgrading the
environment so loaded Python/Pillow libraries are not locked. Restart the app
after installation. The publication runtime uses Pillow 13 and Streamlit 1.66.1;
older Streamlit releases constrain Pillow to the vulnerable pre-13 range.

```bash
python -m pip install -e ".[dev,metrics]"
python -m pip check
mri3d --help
```

The existing CUDA install can remain if it satisfies `requirements.txt`. Use
`requirements-cuda.txt` when configuring a compatible NVIDIA installation.
Data and weights are ignored local artifacts: Git pull does not download the
new experiment's dataset/checkpoints. Existing M4 checkpoints remain supported.

For copied runs, prefer portable runtime metadata:

```bash
mri3d relocate --root .
```

This rewrites recognized project-local JSON references, not model/tensor bytes or
arbitrary external paths. In a new cache, the VQ path is relative to its manifest.
Diffusion resume compares the substantive manifest rather than an obsolete
developer-machine path.

To share inference-only weights locally without training-state overhead:

```bash
mri3d export-inference --vq <vq-checkpoint> --diffusion <diffusion-checkpoint> --output runs/inference-export
```

Keep full checkpoints for resume. The compact pair has its own checked hashes
and retains links to the original checkpoint hashes; raw/EMA selection is explicit.

## Local controlled comparison

```bash
python scripts/download_mrnet_subset.py --output data/mrnet/MRNet-research-v2 --train 256 --validation 60 --planes sagittal --workers 8 --max-mib 1024
mri3d-study
python scripts/sampling_ablation.py
python scripts/summarize_study.py
```

The study writes a frozen `plan.json` before training. Both arms use the same
cohort, seeds, batch sizes and optimizer-update budgets. The combined upgrade is
an exploratory package comparison, not a factorial experiment identifying which
component caused an improvement. The arms predict different noise targets, so
their raw diffusion losses are not directly comparable.

`runs/RESEARCH-V2/study.json` includes per-seed metrics, paired differences and
within-seed paired exam intervals. Cohort manifests record scanner/site metadata
as unavailable rather than inventing subgroup analyses. In the dashboard, set
**Research runs directory** to `runs/RESEARCH-V2` to compare the trained arms.

The sampling ablation holds weights and initial seeds fixed while comparing
raw/EMA weights, DDPM/DDIM and CFG. It also samples an alternate requested label
vector. These diagnostics do not establish the presence of a lesion.

## Independent evaluation workflow

Prepare a separately acquired, consistently preprocessed cohort with a completed
`metadata.csv` and genuine subject linkage where available. Keep the training
cohort frozen. Do not randomly split MRNet slices or relabel previously inspected
validation scans as untouched test images.

```bash
mri3d freeze-protocol --training-data <training-run> --evaluation-data <new-evaluation-run> --split test --output <new-plan.json> --seeds 42 43 44 --hypothesis "Compare the declared reconstruction baselines on new patients"
mri3d evaluate-reconstruction --data <new-evaluation-run> --training-data <training-run> --vq <selected-vq.pt> --split test --protocol <new-plan.json> --output <new-results>
mri3d evaluate-generation --data <new-evaluation-run> --training-data <training-run> --real-split test --generated <generated-run> --output <generation-results.json>
```

The audit checks patient/exam/scan identity plus source/tensor content against
training and model-selection data. Missing patient IDs are reported explicitly;
disjoint exam IDs alone are insufficient to certify independent patients.
Freezing records the plan and timestamp; it cannot prove that an operator never
previously inspected the images. Evaluation receipts prevent silent reuse of a
completed test protocol through this interface.

## Feature metrics and human review

```bash
mri3d evaluate-features --real <real-embeddings.npy> --generated <generated-embeddings.npy> --extractor-id <model-and-weight-version> --output <feature-results.json>
mri3d prepare-review --data <processed-run> --generated <generated-run> --output <review-packet> --count 8
mri3d summarize-review --packet <review-packet> --scores <actual-completed-ratings.csv> --output <review-results.json>
```

Feature arrays must represent the declared exam-level protocol, not correlated
slices counted as independent patients. The extractor, preprocessing, weights,
sample size and uncertainty require justification. Small-sample covariance is
flagged; unbiased KID can legitimately be negative.

The blinded packet contains pseudonymous cases, full normalized tensors, preview
images, instructions, an empty rating template and a separate unblinding key.
The summarizer rejects blank/incomplete ratings and requires at least two reviewers.
Reviewer qualifications and clinical validity cannot be verified by this software.
No radiologist scores or segmentation masks are generated as stand-ins for real evidence.

## Quality checks

Local publication verification: **124 tests passed** (no skips here), **87.82%**
combined statement/branch coverage, Ruff and focused mypy passed, and `pip check`
reported no broken requirements. Actual CUDA tests exercised the upgraded path.
Chrome passed the existing **20 workflow scenarios** and two additional checks
for the new models, relative artifacts, tokenizer statistics and live EMA/CFG/DDIM
sampling. The independent-cohort path was exercised with explicitly artificial
software fixtures, not represented as a new real-world test result.

```bash
python -m ruff check .
python -m mypy
python -m pytest -q --cov --cov-report=term-missing
```

Ruff covers source/scripts/tests. Type checking initially targets the artifact and
statistical helpers; this is not full-project static typing. Coverage measures
statements and branches and enforces a 70% floor. Numerical fixtures exercise
resume equivalence, metric references, overlap, corruption, geometry invariants,
guidance, sampling and EMA. GPU tests also exercise the new mixed-precision path
when CUDA is actually available; CPU CI skips hardware-only cases explicitly.

## What is intentionally not claimed as solved

- Independent patient-disjoint test performance: needs trustworthy new data.
- Expert-confirmed anatomy/pathology or downstream benefit: needs completed studies.
- Full MRNet/site/scanner representativeness: 256/60 remains a reduced cohort.
- Dataset redistribution permission: mirror metadata does not replace upstream terms.
- Clinical deployment, multi-user hosting, or medical certification: outside this local research revision.
- Residual VQ, continuous-VAE comparison, perceptual networks, distributed training,
  DICOM/BIDS workflows, orchestration services and cloud artifact stores: additional
  research/platform directions, not automatic prerequisites or guaranteed quality fixes.
