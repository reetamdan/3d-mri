# M4 reduced-cohort baseline

Recorded on **21 October 2026**. This summary contains measured aggregate results;
raw scans, selected exam lists, checkpoints, local paths, and complete run manifests
are retained in local output directories.

## Dataset and protocol

- Source: public [`cjinny/mrnet-v1`](https://www.kaggle.com/datasets/cjinny/mrnet-v1)
  Kaggle mirror, version 1; upstream dataset is [Stanford MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/).
- Mirror license metadata: unspecified. No new data license is asserted.
- Cohort: 96 official-training exams and 24 official-validation exams.
- Acquisition: all three planes, 360 source arrays, approximately 694.3 MiB.
- Selection: seeded proportional joint-label strata, separately within official splits.
- Training series: sagittal, resized to 128×128×32 and normalized to [-1,1].
- No independent hidden-test evaluation; validation also selects checkpoints.

| Positive exam labels | Train | Validation |
|---|---:|---:|
| Abnormal | 78 | 19 |
| ACL tear | 18 | 11 |
| Meniscal tear | 34 | 10 |

## Training

Hardware/software: RTX 5070 Ti Laptop GPU, Python 3.12.14, PyTorch 2.11.0+cu128,
CUDA 12.8. Recorded training-loop durations exclude download and setup.

| Model | Epochs | Optimizer updates | Selected epoch | Training-loop seconds |
|---|---:|---:|---:|---:|
| 3D VQ-VAE | 40 | 1,920 | 38 | 111.6 |
| Conditional latent DDPM | 300 | 14,400 | 283 | 455.3 |

The VQ model uses eight latent channels and 128 codebook vectors. The DDPM uses
a conditioned 3D U-Net, bottleneck spatial/depth attention, and 200 diffusion steps.
Sampling bounds predicted-clean latents to the trained codebook's channel envelope.
Earlier unstable sampling outputs were retained locally as ablations.

## Internal validation reconstruction

| Metric | Trained VQ | Half-resolution interpolation reference |
|---|---:|---:|
| MSE ↓ | 0.01537 | 0.02975 |
| MAE ↓ | 0.07493 | 0.08920 |
| PSNR ↑ | 24.29968 dB | 21.36067 dB |
| SSIM ↑ | 0.76587 | 0.77707 |

The learned reconstruction improves intensity errors and PSNR but has slightly
lower SSIM. The interpolation reference has a different information budget and
is not an equivalent learned-compression system.

## Generation and limitations

Eight saved samples produced pairwise adaptive 3D MS-SSIM **0.22963**, versus
**0.30247** for eight selected real training references (window 3, four scales).
Lower similarity can reflect artifacts as well as diversity. Same-device,
same-seed sampling reproduced identical bytes; live inference with a changed
seed produced a distinct latent.

Generated volumes remain blurry and do not reliably reproduce coherent large-scale
knee anatomy. Requested pathology conditions have not been expert verified. This
run demonstrates trained modules and reproducible inference, not realistic MRI
synthesis, downstream benefit, clinical utility, or full-cohort generalization.

## Reproduce the full local evidence

```bash
python scripts/download_mrnet_subset.py --train 96 --validation 24 --max-mib 1024
python scripts/run_m4.py
```

The completed run writes `evidence.json`, `evidence.md`, training curves,
validation comparisons, and generated-volume plots under its ignored run folder.
Use the [M4 demo guide](../milestones/m4-demo.md) for setup and interactive inference.
