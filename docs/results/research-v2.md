# Research v2 — controlled local comparison

Measured on the same 256-train / 60-validation sagittal MRNet cohort at 128×128×32.
Seeds: 42, 43, 44.

**Exploratory internal validation, not an independent patient-disjoint test.**

Both arms received the same declared optimizer-update and batch-size budgets. The upgraded arm combines
a larger EMA tokenizer, multiscale denoiser, weight EMA, v-prediction, mixed precision and label dropout.
It uses 50-step DDIM with guidance 1.5 for sampling; the control uses full DDPM.
This comparison cannot attribute gains to one component, and training noise losses across epsilon/v targets are not directly comparable.

## Reconstruction: mean ± sample standard deviation across training seeds

| Arm | MSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| baseline | 0.01545 ± 0.00029 | 0.07470 ± 0.00083 | 24.24292 ± 0.08266 | 0.76283 ± 0.00661 |
| upgraded | 0.01254 ± 0.00022 | 0.06702 ± 0.00053 | 25.17329 ± 0.07948 | 0.79809 ± 0.00576 |
| Interpolation reference | 0.03067 | 0.09038 | 21.20469 | 0.76958 |

The interpolation reference is measured on these same 60 tensors. Its information budget differs from the learned codebook.

## Every seed

| Arm | Seed | PSNR | SSIM | Used codes | Dead-code fraction |
|---|---:|---:|---:|---:|---:|
| baseline | 42 | 24.21633 | 0.76178 | 128 | 0.00000 |
| upgraded | 42 | 25.17818 | 0.80084 | 256 | 0.00000 |
| baseline | 43 | 24.17682 | 0.75681 | 128 | 0.00000 |
| upgraded | 43 | 25.25021 | 0.80196 | 256 | 0.00000 |
| baseline | 44 | 24.33559 | 0.76991 | 127 | 0.00781 |
| upgraded | 44 | 25.09148 | 0.79147 | 256 | 0.00000 |

Within-seed paired exam-bootstrap intervals are recorded in `study.json`;
patient IDs are unavailable, so these are exam-level intervals and do not account for unidentified repeat patients.
Three seeds provide limited evidence about training variability; intervals do not convert validation into a test set.

## Model and sampling diagnostics

- baseline: 214,697 VQ parameters and 591,880 denoiser parameters.
  Memorization search covers all 256 recorded training exams, independently of the diversity sample cap.
- upgraded: 836,937 VQ parameters and 3,474,632 denoiser parameters.
  Memorization search covers all 256 recorded training exams, independently of the diversity sample cap.

All sample-level quantization distances, clipping trajectories, nearest raw/pooled-feature neighbours,
class-matched validation reference counts and texture descriptors are in each generation report.
The 4×4×4 pooled descriptor is not a validated medical perceptual embedding.

## Limits of the result

- Anatomical/pathology fidelity and clinical utility are not established by these metrics.
- No independent test cohort or qualified expert-review results were available for this study.
- Synthetic-label fidelity and downstream augmentation benefit remain unproven.
- Feature FID/KID tooling requires a justified external extractor and sufficient samples; no medical FID is invented.
- The mirror license is unspecified; upstream data-use terms still apply.

## Reproduce

```bash
python -m pip install -e ".[dev,metrics]"
python scripts/download_mrnet_subset.py --output data/mrnet/MRNet-research-v2 --train 256 --validation 60 --planes sagittal --workers 8 --max-mib 1024
mri3d-study
python scripts/sampling_ablation.py
```

Data, checkpoint and sample artifacts remain local. Git contains source, configurations and aggregate results.
