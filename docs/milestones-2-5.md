# Milestones 2–5: model and research workflows

## Current state

Implemented and tested: MRNet import, a 3D VQ-VAE with optional patch-adversarial
VQ-GAN training, a conditioned 3D U-Net/DDPM, latent caching, random-noise
sampling and decoding, reconstruction/generation evaluation, and matched-seed
classification augmentation experiments. The dashboard has **MRNet Dataset**
and **Model Experiments** pages.

The college M1–M4 implementation now includes real MRNet reduced-cohort GPU
training, validation-selected checkpoints, held-out reconstruction, generation,
computed evidence, and live trained inference. See [the assessment pack](milestones/README.md)
and `runs/latest_m4.json`. Historical two-epoch Slicer brain runs remain under
`runs/SMOKE_TEST-*` and are excluded from research results. The current knee run
is an academic baseline, with anatomical review and downstream benefit still unproven.

The Model Experiments dashboard excludes smoke-test artifacts and legacy runs
without checkpoint integrity records. It shows an empty state until a completed
research VQ run exists under `runs/<experiment>/vq`. Checksums establish file
integrity, not model quality. Research training requires a validation split.
Processed tensors, latent caches, generated volumes and paired reconstructions
now have content checksums. Dataset fingerprints include tensor bytes as well
as metadata. Older caches/checkpoints use incompatible fingerprints: reprocess,
train and evaluate again rather than relabeling old test artifacts.

Stanford's official data page is
https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c and currently requires
**Apply for access**. A real subset was obtained separately from the public
`cjinny/mrnet-v1` Kaggle mirror, version 1, using the explicit downloader.
The source manifest records its provenance and unspecified mirror-license
metadata; this is not a claim of official Stanford access approval.

The machine has an NVIDIA RTX 5070 Ti Laptop GPU with approximately 12 GiB VRAM,
and the publication environment uses patched PyTorch 2.14.1+cu130 with CUDA 13.0.
Historical training used 2.11.0+cu128 and retains those versions in its evidence.
The code supports `device: auto`,
`cpu`, or `cuda`; explicit CUDA requests fail clearly if unavailable. The smoke
configuration uses CPU. The M4 run records actual optimizer steps, duration,
and peak allocated CUDA memory in its generated evidence. Larger experiments
still require measurement; baseline execution does not establish convergence.

## MRNet data contract

Stanford describes official patient-disjoint train and validation splits, with
the test set hidden. Preserve those splits: do not create random slice or exam
splits. Public arrays lack patient IDs and spatial affines, so this adapter trusts
the official split assignment rather than inventing patient identities.

The adapter expects the conventional extracted MRNet v1 layout:

```text
MRNet-v1.0/
  train/axial/0000.npy
  train/coronal/0000.npy
  train/sagittal/0000.npy
  valid/axial/1130.npy
  valid/coronal/1130.npy
  valid/sagittal/1130.npy
  train-abnormal.csv
  train-acl.csv
  train-meniscus.csv
  valid-abnormal.csv
  valid-acl.csv
  valid-meniscus.csv
```

CSV files are headerless `exam_id,0_or_1` records. Each NPY is a scalar
`[slice,row,column]` series. Real Kaggle-mirror MRNet files have been downloaded,
prepared, trained and tested using this layout. The official Redivis delivery may
need rearrangement after approval; that separate archive-delivery format has not
been integrated. Artificial layout fixtures additionally cover software edge cases.

The three acquired planes can differ in sequence/contrast and are not registered
views of one isotropic volume. Train separate models per selected series. The
MRNet viewer shows the actual acquired axial, coronal, and sagittal series.
It does not fabricate millimeter spacing or synchronize unrelated slices.
Processed MRNet tensors are `[C,row,column,slice]`, normalized to [-1,1]. The
3D convolutional models use generic spatial axes; no NIfTI affine is assigned.

## Reproducible commands

Run from the repository root with the virtual environment activated. Every
output directory must be new, except explicit checkpoint resume. Commands below
are research recipes, not a statement that these experiments have completed.

### 1. Prepare a manageable subset

```bash
python scripts/research.py prepare-mrnet --root data/mrnet/MRNet-v1.0 --output data/processed/mrnet-sagittal --plane sagittal --shape 128 128 32 --limit 100
```

`--limit` caps exams in each official split, selected in stable ID order; this
is a convenience subset, not a stratified or representative cohort. Review class
counts, use a predeclared cohort for formal research, and retain all repeat exams
within the official patient-disjoint assignment. Invalid/missing scans are logged
and valid scans continue. No center crop discards the field of view; resizing can
still distort resolution, so review anatomy and document the parameter choice.

### 2. Train VQ-VAE or VQ-GAN

```bash
python scripts/research.py train-vq --data data/processed/mrnet-sagittal --output runs/knee/vq --config configs/models.yaml
python scripts/research.py evaluate-reconstruction --data data/processed/mrnet-sagittal --vq runs/knee/vq/last.pt --output runs/knee/reconstruction
```

The encoder downsamples each spatial axis by 4. A trainable vector codebook uses
nearest-neighbor assignments, codebook loss, and commitment loss, with the
straight-through estimator for encoder gradients. The decoder uses transposed
3D convolutions and tanh output. Set `gan_weight: 0` for VQ-VAE; a positive weight
enables a 3D patch discriminator, hinge discriminator loss, and adversarial
generator loss after `gan_start_epoch`. This is a compact research baseline,
without a pretrained perceptual network or a claimed reproduction of the paper.

The training log records L1, codebook loss/perplexity, discriminator loss, and
validation L1. Watch for codebook collapse and oversmoothing. Paired evaluation
saves reconstruction arrays and MSE/MAE/SSIM/PSNR on the same normalized grid.
The dashboard can compare original input tensors and reconstructions. Metrics
alone do not establish ligament, cartilage, or meniscus preservation.

Checkpoints contain model/optimizer states, epoch, random-generator states,
configuration, input manifest fingerprint, and generator-training scan IDs.
Increase total `epochs` and add `--resume` to continue the same run. Configuration
or dataset mismatches are rejected. Exact reproducibility is intended within a
fixed software/hardware environment, not across CPU/GPU implementations.

### 3. Cache latents and train conditional diffusion

```bash
python scripts/research.py cache-latents --data data/processed/mrnet-sagittal --vq runs/knee/vq/last.pt --output runs/knee/latents
python scripts/research.py train-diffusion --data runs/knee/latents --output runs/knee/diffusion --config configs/models.yaml
```

The frozen encoder generates quantized latent tensors. A scalar mean/std is
fitted on training latents only and reused for validation/test and sampling.
The cache records the exact VQ checkpoint checksum. Diffusion uses a cosine beta
schedule, forward Gaussian noise, and epsilon-prediction MSE. The 3D U-Net uses
sinusoidal timestep embeddings, conditional residual blocks, and skip connections.
The M4 configuration additionally enables bottleneck spatial/depth self-attention.
`conditional: true` requires genuine `[abnormal,acl,meniscus]` labels. It never
assigns invented labels to unlabeled brain data. Epoch logs include training
noise MSE and validation MSE with fixed validation noise draws.

This is full-step DDPM sampling with the posterior variance and predicted-clean
latent bounds derived from the trained codebook's per-channel extrema. The
initial unbounded protocol exhibited extreme latent amplitudes and is retained
as a sampling ablation. Bounding a sample is not proof of anatomical fidelity.
The historical M4 configuration does not use EMA, guidance or DDIM. Research v2
adds these as explicit options, including multiresolution denoising, v-prediction,
packed latent caches and matched-seed comparisons; see [the upgrade guide](research-upgrade.md).
A decreasing denoising loss alone does not prove convergence or anatomical quality.

### 4. Generate and evaluate

```bash
python scripts/research.py generate --vq runs/knee/vq/last.pt --diffusion runs/knee/diffusion/last.pt --output runs/knee/generated --count 8 --labels 1 1 0 --seed 42
python scripts/research.py evaluate-generation --data data/processed/mrnet-sagittal --generated runs/knee/generated --output runs/knee/generation-report.json
```

Sampling starts from independent seeded Gaussian noise. Latents are denormalized,
snapped to the VQ codebook, and decoded. The manifest records conditioning,
seeds, model hashes, and generator training lineage. Requested disease conditions
are not verified clinical labels. Choose clinically plausible conditioning
combinations represented in the training cohort and inspect class fidelity.

Unpaired generation diagnostics include pairwise MS-SSIM diversity, intensity
ranges, per-axis adjacent-slice MAE, and raw/pooled-spatial nearest-training
distances as limited memorization screens. The updated search covers the full
recorded generator training cohort, separately from the diversity sample cap.
Real and generated diversity use the same geometry/protocol; selection and class
matching are reported. Updated fields distinguish the real comparison split.
The internal MS-SSIM implementation uses a Gaussian window of 3 and up to five
scales, truncating and renormalizing standard weights for thin volumes. The
actual number of scales is reported; these values should not be directly compared
with a paper using a different window/scale protocol. Tests establish basic
identity and differing-input behavior. Research v2 additionally checks single- and
five-scale values against `pytorch-msssim` under a matching numerical protocol;
that does not independently validate medical interpretation of the metric.

SSIM/PSNR are reported for paired VQ reconstructions, not arbitrary generated vs
real pairs. Lower diversity MS-SSIM can indicate noise as well as diversity;
adjacent-slice differences are not an anatomical validity score. Visual review
must explicitly examine artifacts, continuity, plausible structures, and near
copies of training images. Generated volumes have no patient-space affine.

### 5. Compare downstream augmentation

MRNet has exam-level abnormality, ACL, and meniscus labels. It does not supply the
segmentation masks needed for a Dice-based study, so the implemented downstream
task is three-label research classification with AUC, accuracy, and Brier score.
Dice/IoU helpers are provided for separately obtained aligned masks; no masks or
segmentation results are fabricated.

```bash
python scripts/research.py compare-augmentation --data data/processed/mrnet-sagittal --generated runs/knee/generated --output runs/knee/downstream --limit 100 --steps 1000 --seeds 42 43 44
```

Two compact 3D classifiers use identical initial seeds and optimizer-update
budgets. The real-only arm draws real scans; the augmented arm alternates real
and synthetic draws. All evaluation scans are real. The report saves individual
seed metrics and matched-seed augmented-minus-real differences. AUC is unavailable
when one class is absent, rather than reported as zero.

The generator must have seen only the same limited real-training cohort used by
the classifiers; the experiment rejects outside-cohort lineage. Use the same
`limit` when preparing/training the generator and downstream experiment. Do not
train a generator on a full cohort and describe the classifier experiment as
having access to only a small subset. Validation is the official public split,
not an independent test set. This baseline uses fixed training budgets and no
validation-driven checkpoint selection.

The generation command creates one conditioning combination per run. A full
study needs a predeclared conditioning/class balance strategy; one pool cannot
establish improvement for all classes. Synthetic condition labels require expert
verification. Repeated seeds and descriptive differences are not sufficient for
a clinical utility or significance claim.

## Research gap and work still required

The proposed gap is whether limited-data, conditionally generated knee volumes
improve a downstream task without degrading anatomical fidelity or copying
training subjects. This is a hypothesis, not a demonstrated novelty claim.
For expanded research: obtain the intended full dataset through its applicable
access process, freeze cohort and preprocessing, extend and compare training,
review reconstructions and generated anatomy,
predeclare augmentation balance and fixed-budget controls, estimate uncertainty,
and evaluate on an independent patient-disjoint test set when available.

Longitudinal generation would need linked timepoints and registration beyond
the current MRNet adapter. Future comparisons can investigate stronger denoisers,
perceptual/adversarial losses, conditional fidelity, held-out anatomical review,
and segmentation only with appropriate mask annotations.

Sources: Stanford dataset description and patient split policy:
https://stanfordmlgroup.github.io/competitions/mrnet/ ; model task description:
https://stanfordmlgroup.github.io/projects/mrnet/ ; access portal:
https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c .
