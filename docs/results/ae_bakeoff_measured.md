# Measured autoencoder comparisons

Recorded measurements on the frozen 96/24 training/validation cohort.
Means ± sample SD describe training seeds; single-seed configurations are marked explicitly.

| Arm | Seeds | Spatial compression | PSNR dB | SSIM | MS-SSIM | Gradient ratio | HF ratio | Peak train MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ae_kl_aniso | 42,43,44 | 4×4×2 | 26.802 ± 0.199 | 0.8578 ± 0.0092 | 0.9696 | 0.467 | 0.160 | 775.8 |
| ae_kl_aniso_deconv | 42 | 4×4×2 | 26.463 (one seed) | 0.8463 (one seed) | 0.9676 | 0.459 | 0.134 | 741.4 |
| ae_kl_aniso_perceptual | 42 | 4×4×2 | 26.891 (one seed) | 0.8625 (one seed) | 0.9781 | 0.485 | 0.149 | 1052.9 |
| ae_kl_aniso_ssim | 42 | 4×4×2 | 27.003 (one seed) | 0.8734 (one seed) | 0.9796 | 0.501 | 0.145 | 807.9 |
| ae_kl_aniso_structural | 42 | 4×4×2 | 26.981 (one seed) | 0.8745 (one seed) | 0.9790 | 0.510 | 0.157 | 809.1 |
| ae_kl_iso | 42,43,44 | 4×4×4 | 25.502 ± 0.070 | 0.8169 ± 0.0037 | 0.9554 | 0.358 | 0.097 | 751.8 |
| ae_kl_low_compression | 42 | 2×2×2 | 29.072 (one seed) | 0.9158 (one seed) | 0.9863 | 0.570 | 0.255 | 918.4 |
| ae_vq_baseline | 42,43,44 | 4×4×4 | 24.352 ± 0.100 | 0.7691 ± 0.0081 | 0.9349 | 0.314 | 0.108 | 202.9 |
| ae_vq_improved | 42 | 4×4×4 | 24.640 (one seed) | 0.7864 (one seed) | 0.9425 | 0.335 | 0.111 | 202.9 |

Gradient and frequency measurements use the common resized array grid.
Continuous latent scalar counts and discrete-code bitrates describe different representations.

## Every completed run

| Arm | Seed | PSNR | SSIM | Updates | Effective batch |
|---|---:|---:|---:|---:|---:|
| ae_vq_baseline | 42 | 24.41399 | 0.77366 | 1920 | 2 |
| ae_vq_improved | 42 | 24.64043 | 0.78636 | 1920 | 2 |
| ae_kl_iso | 42 | 25.54694 | 0.81910 | 1920 | 2 |
| ae_kl_aniso | 42 | 27.02920 | 0.86819 | 1920 | 2 |
| ae_kl_aniso_ssim | 42 | 27.00255 | 0.87341 | 1920 | 2 |
| ae_kl_aniso_structural | 42 | 26.98093 | 0.87453 | 1920 | 2 |
| ae_vq_baseline | 43 | 24.23599 | 0.75967 | 1920 | 2 |
| ae_kl_iso | 43 | 25.53819 | 0.81901 | 1920 | 2 |
| ae_kl_aniso | 43 | 26.71779 | 0.85434 | 1920 | 2 |
| ae_vq_baseline | 44 | 24.40465 | 0.77390 | 1920 | 2 |
| ae_kl_iso | 44 | 25.42167 | 0.81264 | 1920 | 2 |
| ae_kl_aniso | 44 | 26.65957 | 0.85085 | 1920 | 2 |
| ae_kl_aniso_perceptual | 42 | 26.89135 | 0.86247 | 1920 | 2 |
| ae_kl_low_compression | 42 | 29.07243 | 0.91578 | 1920 | 2 |
| ae_kl_aniso_deconv | 42 | 26.46336 | 0.84628 | 1920 | 2 |

Research project only. Not for real medical use.
