# Recorded M4 output study

The 22 October 2026 study examines the local 96-training / 24-validation M4
experiment. Its saved measurements are retained in
`runs/OUTPUT-VERIFICATION-20261022-03/`.

## Data and model records

The study covers 360 source arrays, 120 processed tensors, 120 cached latents,
24 reconstructions and eight saved diffusion samples. Raw-to-processed replay
has zero maximum absolute difference. Saved reconstruction metric differences
are at most 1.32e-13. The selected VQ epoch is 38/40, diffusion 283/300, with
1,920 and 14,400 optimizer updates respectively.

## Reconstruction

| Metric | VQ | Interpolation | Train-mean image |
|---|---:|---:|---:|
| MSE | .015368 | .029747 | .090409 |
| MAE | .074935 | .089197 | .209265 |
| PSNR dB | 24.29968 | 21.36067 | 16.49929 |
| SSIM | .76587 | .77707 | .30274 |

Paired exam-bootstrap differences for VQ minus interpolation are PSNR +2.939 dB
[2.487,3.341] and SSIM −.01120 [−.01628,−.00635], using 10,000 resamples.
The tokenizer uses 125/128 codes with perplexity 70.27 and a 2.34% zero-use fraction.

## Sampling records

Eight sample seeds produce distinct latent files; repeated sampling with the same
runtime and seed produces identical image/latent bytes. Cross-environment replay
between the historical 5070 Ti/torch 2.11 and 3060/torch 2.14 has mean code
agreement 99.962%, maximum latent MSE 2.50e-5 and current reconstruction
PSNR/SSIM 24.29983 / .76588.

Adaptive pairwise MS-SSIM is .22963 for generated outputs and .32260 for the
selected real-training references used in this study. Mean generated gradient
energy is 16.1% of the real-training mean. Full-training nearest pixel MSE ranges
.11932–.14728 across the eight outputs. Condition inputs and all per-sample
measurements are stored with the local manifests.

## Historical augmentation measurements

Three seeds, 400 optimizer updates per arm, same 96/24 cohort:

| Task · validation AUROC | Real only | Real + synthetic | Label-matched resampling |
|---|---:|---:|---:|
| Abnormal | .7614 | .5860 | .7684 |
| ACL | .2517 | .4219 | .2704 |
| Meniscus | .2786 | .3405 | .2857 |

At threshold .5, ACL and meniscus sensitivity are zero. Mean Brier errors are
.3409 / .3663 for real-only / synthetic ACL, and .2577 / .2945 for meniscus.
The eight synthetic conditioning vectors in this probe are `[1,0,0]`.

The recorded values describe this experimental configuration and evaluation
protocol. The main research report lists the subsequent representation study.

Research project only. Not for real medical use.
