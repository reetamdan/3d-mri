# Frozen starting point for the representation study

Recorded before research model changes, 23 October 2026.

- Starting Git commit: `33d55d3` (`Patch image dependencies and update Streamlit APIs`).
- Engineering hardening and the independent output audit are committed separately
  before the representation study; their commit is the research branch's parent.
- Dataset fingerprint: `11f72fdcb313825d05e563feb0b6c97726cc60556acc65e057ae2e1359cbb736`.
- Data: 96 official-training / 24 official-validation MRNet sagittal exams.
  Patient identifiers, laterality, physical affines/spacing and scanner metadata
  are unavailable. Exam separation is not a verified patient-level split.
- VQ checkpoint SHA-256: `9fa30629be9b729e36a0508fd12d2789fa9a663da7614b5c70baee9e6b564fde`.
- Diffusion checkpoint SHA-256: `82e7a8a0df3ef82bf1e23fffa638eab18fd545663f72b2be4699278db220e63a`.
- Exact baseline configurations: `configs/m4-vq.yaml`, `configs/m4-diffusion.yaml`.
- Original local artifacts: `runs/M4-MRNET-20261021/`.
- Independent verification: `runs/OUTPUT-VERIFICATION-20261022-03/` and
  `docs/results/output-verification.md`. Model/data artifacts remain local.

## Architecture map verified from source

| Stage | Actual contract |
|---|---|
| MRNet input | uint8 `[slices,256,256]`, three separately acquired planes |
| Model preprocessing | sagittal only; transpose to `[row,column,slice]`, trilinear resize with `align_corners=False`, per-volume min-max to `[-1,1]`; no crop, pad, physical resampling or inferred orientation |
| Stored/model tensor | `[1,128,128,32]` / `[B,1,128,128,32]` |
| VQ encoder | two stride-2 Conv3d blocks, residual blocks, 8 latent channels |
| Quantized latent | `[B,8,32,32,8]`; 4×4×4 spatial downsampling; 128 codes |
| VQ objective | mean L1 + codebook/commitment MSE (commitment 0.25); no GAN in measured baseline |
| VQ decoder | residual block and two transposed convolutions, final tanh |
| Latent normalization | one global mean/std fitted on training quantized latents, saved with lineage |
| Denoiser | small conditioned 3D U-Net; time embedding plus global three-label linear embedding injected in TimeBlocks; bottleneck attention |
| Baseline diffusion | cosine 200-step epsilon DDPM, Adam, validation-selected weights; no weight EMA or label dropout in the measured baseline |
| Baseline sampling | all DDPM steps, per-channel predicted-clean clipping from trained codebook bounds, codebook projection before decoding |
| Existing optional upgrades | multiscale U-Net, EMA, v-prediction, learned null labels/dropout, CFG, DDIM, gradient accumulation and mixed precision |
| Downstream | compact three-label Conv3d classifier; matched-update real/synthetic comparisons; current 0.5 operating point predicts majority classes |
| Evaluation | paired reconstruction metrics; separate unpaired generation diagnostics; full-training nearest-neighbor screening; explicit content/identity audits |

The NIfTI workflow is separate: canonical RAS, unit-aware physical resampling,
affine-preserving crop/pad/resize, configurable normalization and subject grouping.
Its anatomical-axis semantics must not be applied to MRNet arrays.

## Measured starting performance

| Metric | VQ | Interpolation |
|---|---:|---:|
| MSE | 0.015368 | 0.029747 |
| MAE | 0.074935 | 0.089197 |
| PSNR | 24.29968 dB | 21.36067 dB |
| SSIM | 0.76587 | 0.77707 |

The recorded VQ/interpolation comparison has higher VQ PSNR on 23/24 exams and
higher VQ SSIM on 4/24. Generated gradient energy is 16.1% of the training mean.
The three-seed classification probe records ACL/meniscus sensitivity of zero at
threshold .5. Additional measurements are in the research report.

## First representation question

Compare the same frozen inputs/exams and optimizer-update budgets using a compact
KL autoencoder at 4×4×4, then 4×4×2 compression, then a structural objective.
Also retain a matched-budget existing VQ control and a small improved VQ control.
Record the protocol before training and measure pixels, structure, edges and
frequency content on the same inputs.

Research project only. Not for real medical use.
