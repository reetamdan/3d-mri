# Model configuration card

## Project scope

Local research into 3D knee MRI representations, reconstruction and latent
sampling. The website exposes the corresponding image tools and recorded settings.

## Data

The representation study uses 96 official-training and 24 official-validation
MRNet sagittal exams, resized to 128×128×32 and min-max normalized to [-1,1].
The historical 256/60 comparison is a separate recorded experiment in
`results/research-v2.md`. Patient linkage, physical affines and complete scanner
metadata are not supplied by these arrays. Dataset terms are separate from the
software; data and weights are local artifacts.

## Models

- Native VQ: compact two-downsample encoder, eight latent channels, 128 codes,
  L1 plus VQ regularization, transposed-convolution decoder.
- VQ capacity configuration: 256 codes with the matched training budget.
- MONAI KL: channels 8/16/32, eight latent channels, isotropic 4×4×4 or
  anisotropic 4×4×2 sampling, sampled training and posterior-mean reconstruction.
- Objective configurations: L1, adaptive MS-SSIM, three-axis gradients, explicit
  frozen medical features and the corresponding latent regularizer.
- Diffusion: time/label-conditioned 3D U-Net, cosine scheduler, epsilon/v targets,
  optional EMA, label dropout, CFG, Min-SNR and condition-token attention.
- Sampling: recorded DDPM/DDIM settings and training-fitted latent scaling/bounds.

## Measurements

Reconstruction uses per-exam MSE, MAE, PSNR, SSIM, adaptive MS-SSIM and unit-grid
edge/frequency descriptors. Sampling studies record seeds, condition inputs,
latent values, sample statistics and training-reference similarities.
`results/ae_bakeoff_measured.md` and `research_v2_report.md` contain the recorded values.

## Reproducibility

Configurations are in `configs/`. Checkpoints, cohort identities, normalization
constants, source hashes and selected epochs are stored with local run records.
The application checks model/data/image relationships before display.

Research project only. Not for real medical use.
