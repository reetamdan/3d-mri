# Research v2 model card

## Intended use

Local research into reproducible knee-MRI compression and latent generative
modelling. The models support controlled experiments, visualization and teaching.
They are not diagnostic models, certified anatomical simulators, or a source of
expert-verified disease examples.

## Data

The controlled study uses a seeded joint-label-stratified subset of public
MRNet-mirror data: 256 official-training and 60 official-validation exams,
sagittal series only, resized to 128×128×32 and min-max normalized to [-1,1].
The official split is retained. The arrays do not provide reliable patient IDs,
patient-space affines or complete scanner/site metadata. Prevalence and sequence
variation can differ between cohorts. The mirror's license metadata is unspecified;
the original dataset's terms govern usage. Raw data are not redistributed here.

## Architectures and objectives

The control retains the compact two-downsample VQ-VAE and one-down/up latent
U-Net. The upgrade increases VQ width/codebook capacity, optionally updates the
codebook with EMA and dead-code resets, and uses a multiresolution conditioned
3D U-Net with bottleneck attention. Denoising supports epsilon/v prediction,
EMA inference weights, label dropout and classifier-free guidance. DDPM and
DDIM use explicit, recorded inference settings. Predicted-clean latent clipping
uses trained codebook bounds and is recorded in diagnostics.

## Evaluation

Per-exam MSE, MAE, PSNR and SSIM measure paired normalized reconstructions.
Foreground diagnostics use an intensity mask, not a segmentation annotation.
Seed-level variability and exam/subject bootstrap intervals are reported with
their actual unit. MRNet's absent patient linkage limits exam-level uncertainty.
Nearest-neighbour checks search all generator training references. Diversity,
texture descriptors and spatial pooled features are not anatomical-quality scores.

The controlled experiment uses internal validation for selection/reporting.
It is not independent test performance. New-cohort overlap audits and frozen
protocol support are available, but cannot create trustworthy new patients or
prove that a dataset was never inspected.

## Known limitations

The [three-seed controlled comparison](results/research-v2.md) reports measured
internal-validation improvements over the matched control and the same-cohort
interpolation reference. That result is separate from an independent anatomical
or clinical validation study.

- Generated structures can remain blurry, inconsistent or anatomically implausible.
- Requested exam-label conditions are not verified pathology in generated images.
- No expert review, segmentation evaluation or downstream utility is inferred
  from software tests or pixel similarity.
- Small cohorts and few training seeds limit precision and generalization.
- Additional losses, attention or larger models do not guarantee clinical fidelity.

See [the upgrade guide](research-upgrade.md) for implemented safeguards, protocols,
operator commands and remaining evidence requirements. Full datasets, weights and
sample manifests remain local; the repository publishes aggregate measured results.
