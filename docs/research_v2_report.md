# 3D knee MRI representation and sampling study

## Project

This study explores learned 3D knee MRI representations, reconstruction objectives,
latent diffusion and classification augmentation. The implementation uses PyTorch,
official MONAI components and config-driven experiments on a laptop GPU.

## Dataset and preprocessing

- 96 official-training and 24 official-validation sagittal MRNet exams.
- 360 source arrays across three acquisitions; 120 processed model tensors.
- Input layout `[B,1,row,column,acquired-series slice]`, size 128×128×32.
- Source `[slice,row,column]` arrays are transposed and trilinearly resized with
  `align_corners=False`, followed by volume-wise min-max scaling to [-1,1].
- Original acquired slice counts range from 17 to 49.
- Physical spacing, laterality and patient linkage are not supplied by these
  released arrays. The study uses recorded exam assignments and content auditing.
- The separate NIfTI workflow uses its supplied spatial affine and units.

Architecture, checkpoint hashes and the cohort fingerprint are recorded in
`docs/current_baseline.md`. External source inspection and citations are in
`docs/external_repos_review.md`.

## Representation experiment design

Each run uses 40 epochs × 48 optimizer updates = **1,920 updates**, effective
batch two, Adam 3e-4 and gradient clipping one. Checkpoints use validation L1
selection. The native VQ uses full precision; KL uses mixed precision.

MONAI KL uses channels 8/16/32, eight latent channels, GroupNorm four groups,
sampled training and posterior-mean reconstruction with a tanh output. Isotropic
sampling is 4×4×4; anisotropic sampling is 4×4×2. The primary KL models have
196,409 parameters. The matched VQ models have 214,697 / 215,721 parameters for
128 / 256 codes.

KL regularization is the mean over latent entries, weighted by 1e-6. This differs
from the spatial-sum reduction in the MONAI tutorial. Structural weights are
fitted once from the first training batch's initial output-gradient norms, with
MS-SSIM and gradient terms each targeted to 25% of the L1 gradient norm.
Adaptive 3D MS-SSIM uses window three; derivatives use the common unit array grid.

## Recorded representation measurements

Three-seed values are mean ± sample SD; other rows are seed-42 studies.

| Representation | Seeds | PSNR dB | SSIM | Gradient ratio | HF ratio |
|---|---|---:|---:|---:|---:|
| Native VQ, 128 codes | 42/43/44 | 24.352 ± .100 | .7691 ± .0081 | .314 | .108 |
| KL isotropic | 42/43/44 | 25.502 ± .070 | .8169 ± .0037 | .358 | .097 |
| KL anisotropic | 42/43/44 | 26.802 ± .199 | .8578 ± .0092 | .467 | .160 |
| VQ, 256 codes | 42 | 24.640 | .7864 | .335 | .111 |
| Anisotropic KL + MS-SSIM | 42 | 27.003 | .8734 | .501 | .145 |
| Anisotropic KL + MS-SSIM + gradient | 42 | 26.981 | .8745 | .510 | .157 |
| Structural KL + medical features | 42 | 26.891 | .8625 | .485 | .149 |
| KL, 2×2×2 spatial reduction | 42 | 29.072 | .9158 | .570 | .255 |
| Anisotropic KL, learned upsampling | 42 | 26.463 | .8463 | .459 | .134 |

All per-run values, checkpoint identifiers and recorded intervals are available in
`docs/results/ae_bakeoff_measured.json`. The 2×2×2 model has a latent tensor
`[8,64,64,16]`, containing the same number of float scalars as its input.

The selected seed-42 anisotropic model has PSNR 27.02920 / SSIM .86819. Relative
to the historical checkpoint, its paired exam-bootstrap PSNR difference is
2.729 dB, with interval [2.541,2.921]; the SSIM difference is .1023
[.0962,.1086]. These are exam-level intervals on model-selection validation.

## Compute and latent statistics

Measured hardware: RTX 3060 Laptop, 6 GiB VRAM, approximately 16 GiB host RAM,
torch 2.14.1+cu130 and MONAI 1.6.1. Peak allocated training memory is approximately
203 MiB VQ, 752 MiB isotropic KL, 776 MiB anisotropic KL, 809 MiB structural KL,
1,053 MiB perceptual KL, 918 MiB 2×2×2 KL and 741 MiB learned-upsample KL.

For the selected anisotropic cache, training-channel means range −2.301 to 3.105,
standard deviations .621 to 2.286, global mean/std −.2253 / 2.2341, and mean
posterior sigma .02070. Exact training-fitted scaling constants are saved with
the frozen latent manifest. Both per-channel and legacy scalar scaling are supported.

## Diffusion and conditioning studies

Four bounded single-training-volume experiments use v-prediction, base-16
`[1,2,4]` UNet, a 32-step cosine schedule, EMA .995, Adam 3e-4, batch two,
2,000 updates and four fixed DDIM32 sample seeds.

| Configuration | SSIM to training reconstruction | Normalized latent MSE | Fixed-noise denoising MSE |
|---|---:|---:|---:|
| Unweighted v | .63788 | .32813 | .34110 |
| Min-SNR gamma 5 | .53915 | .41431 | .49118 |
| Array-coordinate v | .66138 | .30715 | .33367 |
| Array coordinates + Min-SNR | .57944 | .37355 | .45759 |

The coordinate variant adds three normalized array-index channels and 1,344
parameters, with shared initial weights/RNG matched to the ordinary model.
An additional artificial conditional-constant experiment uses ±.5 targets and
500 updates. These experiments study the numerical training and conditioning paths.

Implemented diffusion configurations also include epsilon/v prediction, Min-SNR,
condition dropout, CFG, EMA and optional bottleneck condition-token attention.
The preserved M4 experiment supplies the website's saved and live diffusion outputs.

## Medical features and neighborhood analysis

An explicit frozen MedicalNet ResNet10 descriptor is loaded through MONAI using
official weights at revision `deeeaa1e2a342b63e28b61e9d993b63e79c4b437`, SHA-256
`afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605`.
Inputs are permuted to depth/row/column, resized to 32×64×64 and volume z-scored.

The feature study covers six primary reconstructions and eight historical samples
against all 96 training exams, using medical-feature, SSIM and pooled-octant patch
neighbors. Paired cosine distances are 2.23e-5 for native VQ and 7.71e-6 for
anisotropic KL. Class-matched distribution statistics use 37 real and eight
generated vectors in 512 dimensions, with covariance-rank-limited estimation.

## Classification augmentation measurements

The historical probe uses three seeds and 400 updates per arm on the same 96/24
cohort. Mean validation AUROC:

| Task | Real only | Real + synthetic | Label-matched resampling |
|---|---:|---:|---:|
| Abnormal | .7614 | .5860 | .7684 |
| ACL | .2517 | .4219 | .2704 |
| Meniscus | .2786 | .3405 | .2857 |

At threshold .5, recorded ACL and meniscus sensitivity is zero. A separate
framework implements TorchIO, resampling and synthetic arms with equal budgets,
development-set operating thresholds, AUPRC, Brier score, bootstrap intervals and
five-seed defaults. It is an implementation available for additional studies;
the table above describes the completed historical probe.

## Reproduction

```bash
python -m pip install -e ".[dev,metrics,research]"
python -m pip install -r requirements-research.txt
python scripts/representation_study.py --output runs/AE-BAKEOFF-new --device cuda
python scripts/representation_study.py --output runs/AE-CONFIRMATION-new --arms ae_vq_baseline ae_kl_iso ae_kl_aniso --seeds 43 44
```

These commands use the processed cohort and baseline described in
`docs/current_baseline.md`. Run heavy jobs serially. Configurations, raw
experiment records, source hashes, checkpoint lineage and split identities remain
with their local run folders. Compatible local fastMRI/SKM-TEA import and
same-architecture initialization are available for representation-pretraining studies.

Research project only. Not for real medical use.
