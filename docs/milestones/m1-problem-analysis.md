# M1 — Problem Domain Identification and Analysis

**Assessment window:** September 7–11, 2026 · **Weight:** 3 marks

**Document prepared:** October 21, 2026

## 1. Project title

**Reproducible 3D Knee MRI Generation Using Vector-Quantized Latent Diffusion**

## 2. Problem statement

Medical-image research requires diverse, consistently processed data. MRI is
volumetric: adjacent slices represent connected anatomy, so generating slices
independently can create inconsistent structures. Training diffusion directly
on a full 3D volume also places substantial demands on GPU memory and compute.
Access restrictions, differing scan geometries, limited labels, and patient
overlap across splits complicate evaluation further.

This project investigates whether a compact vector-quantized autoencoder can
learn a useful representation of knee MRI and whether a diffusion model trained
in that representation can generate new 3D volumes on available laptop hardware.
The engineering objective is a reproducible, inspectable workflow from real
scans to a trained model and measured outputs. Anatomical realism and downstream
benefit are research questions, not assumptions.

## 3. Domain and context

- **Domain:** medical imaging, computer vision, representation learning, and
  generative machine learning.
- **Users:** students and researchers preparing data, comparing reconstructions,
  inspecting generative models, and studying augmentation.
- **Imaging target:** knee MRI from MRNet, an examination-level dataset with
  abnormality, ACL-tear, and meniscal-tear labels [1].
- **Reference approach:** Medical Diffusion's two-stage compression and 3D
  latent-diffusion strategy [2].
- **Operational setting:** local Python/PyTorch training and a Streamlit research
  dashboard, with files retained on the researcher's machine.

Stanford reports 1,370 exams overall and patient-disjoint training, validation,
and hidden-test assignments. Public training and validation contain 1,130 and
120 exams respectively [1]. Those published dataset counts are context, not
the counts of the local reduced training cohort.

## 4. Why the problem matters

1. A reproducible preprocessing contract avoids comparing incompatible tensors.
2. Subject-disjoint evaluation reduces optimistic results caused by leakage.
3. Learning a latent representation can reduce the spatial cost of 3D diffusion.
4. Joint volumetric modeling can exploit inter-slice context.
5. Synthetic-data research may eventually help limited-data learning, but only
   if useful anatomy, diversity, and privacy properties are independently tested.

## 5. Scope and boundaries

### Included through the M4 college assessment

- Explicit data acquisition with source/selection/checksum records.
- NIfTI ingestion and a separate MRNet NumPy adapter.
- Validation, configurable preprocessing, deterministic split handling, export.
- Acquired-series and tensor-slice visualization.
- A trained 3D VQ baseline, cached latents, and latent DDPM training/sampling.
- Held-out reconstruction metrics, unpaired generation diagnostics, training
  curves, and a reproducible trained-model demonstration.

### Research extensions

The existing classification augmentation comparison can be run separately with
genuine labels and controlled training budgets. Segmentation requires aligned
masks, which MRNet does not provide. Longitudinal prediction requires linked
timepoints and registration beyond the current MRNet interface.

The system does not diagnose patients or establish that generated images are
clinically valid. It does not infer physical voxel spacing from MRNet arrays.

## 6. Domain-specific constraints

| Constraint | Consequence for the design |
|---|---|
| Three MRNet planes are separate acquisitions | Browse actual series; train one selected plane; do not pretend they are registered orthogonal cuts |
| Arrays lack patient identity and affine | Preserve Stanford's official split assignments; use tensor coordinates |
| Small, imbalanced cohorts | Record joint-label sampling and class support; limit generalization claims |
| Approximately 12 GiB laptop GPU | Use a compact model and configurable batch/shape; record measured resources |
| Generative metrics can be misleading | Use paired PSNR/SSIM only for reconstructions; inspect diversity and artifacts separately |
| Source files and models can change | Verify byte-level hashes and lineage at each stage |

## 7. Research questions and success criteria

**RQ1:** Can the preprocessing and training workflow execute reproducibly on the
available hardware with real, split-separated knee volumes?

**RQ2:** How does held-out VQ reconstruction compare with the project's explicit
interpolation baseline on the same normalized grid?

**RQ3:** Can the trained denoiser generate finite volumes, and what do pairwise
diversity, slice-continuity, and nearest-training diagnostics reveal?

M4 software success requires actual checkpoints, held-out results, recorded
optimizer steps, reproducible inference and a live demonstration. The recorded
measurements describe the selected experimental configurations.

## References

1. Stanford ML Group. *MRNet: A Knee MRI Dataset and Competition.*
   https://stanfordmlgroup.github.io/competitions/mrnet/
2. Khader F. et al. *Medical Diffusion: Denoising Diffusion Probabilistic Models
   for 3D Medical Image Generation.* arXiv:2211.03364.
   https://arxiv.org/abs/2211.03364

Research project only. Not for real medical use.
