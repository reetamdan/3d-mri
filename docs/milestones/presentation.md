# M1–M4 review presentation outline

Use the team's actual names, supervisor, department, and institutional template.
Insert the generated PNGs and actual numeric tables from the selected run; do
not use the historical smoke-run numbers. This is a content-ready slide outline.

## Slide 1 — Title and objective

Reproducible 3D Knee MRI Generation Using Vector-Quantized Latent Diffusion.
Objective: implement and evaluate a laptop-feasible volumetric generation workflow.

## Slide 2 — M1: problem and domain

- Volumetric MRI requires inter-slice context.
- Full-resolution 3D generation has significant compute/memory demands.
- Data handling and evaluation must respect scan geometry and patient separation.
- Primary user: imaging researcher; intended use: research experimentation.

## Slide 3 — M2: existing systems

Summarize the comparison table in `m2-planning-justification.md`: GAN → VQ-VAE /
VQGAN → DDPM → latent diffusion → Medical Diffusion. Cite the original papers.
Explain the advantage and limitation motivating each selected component.

## Slide 4 — M2: feasibility and plan

- Existing RTX 5070 Ti Laptop GPU, CUDA verified.
- Reduced, explicitly selected real cohort and compact latent model.
- Existing hardware and open-source software; variable electricity/network cost.
- Stage separation makes training, evaluation, and debugging independently visible.

## Slide 5 — M2: methodology and algorithm relevance

Show the M2 Mermaid architecture. Explain 3D convolution, vector quantization,
timestep-conditioned U-Net, bottleneck attention, DDPM, and paired/unpaired metrics.

## Slide 6 — M3: real data and preprocessing

Show an actual acquired exam in three separate planes. Include source URL,
selected cohort/class counts, official split handling, tensor axes, normalization,
and checksum records. State that MRNet spacing/affines are unavailable.

## Slide 7 — M3: implemented architecture

Show the code-to-design traceability table and 128×128×32 → 8×32×32×8 latent
contract. Distinguish the selected VQ-VAE baseline from optional VQ-GAN mode.

## Slide 8 — M4: training evidence

Insert `evidence-assets/vq-training.png` and `diffusion-training.png`.
Read actual completed/selected epochs, optimizer steps, duration, and memory
from `evidence.json`. Explain internal validation checkpoint selection.

## Slide 9 — M4: held-out reconstruction

Insert `evidence-assets/validation-1.png`. Add the computed learned-VQ versus
interpolation MSE/MAE/PSNR/SSIM table. Describe the recorded settings and the
representations' information budgets.

## Slide 10 — M4: generation and live demo

Insert `evidence-assets/generated.png`. Run **Generate a new volume** with a
changed seed and show the new artifact manifest. Explain requested conditioning,
fixed intensity scale, and same-device seeded repeatability.

## Slide 11 — Measurements and additional studies

- Describe generated/real diversity using the recorded sample protocol.
- State the cohort, resolution and model-selection validation settings.
- Identify which measurements and configurations were included in the study.
- Additional studies can use larger cohorts, spatial annotations and augmentation comparisons.

## Slide 12 — Reproducibility and questions

Show the one-command runner, evidence path, checkpoint hashes, and test result.
Finish with the M1–M4 criterion mapping from this directory's README.

Research project only. Not for real medical use.
