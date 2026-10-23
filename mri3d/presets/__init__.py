"""Configuration resources populated from the canonical configs/ files at build time."""
PRESETS = ('config.yaml', 'models.yaml', 'm4-vq.yaml', 'm4-diffusion.yaml', 'research-baseline.yaml',
           'research-v2.yaml', 'smoke-models.yaml', 'prd-example.yaml', 'real-scans.yaml',
           'ae_vq_baseline.yaml', 'ae_vq_improved.yaml', 'ae_kl_iso.yaml', 'ae_kl_aniso.yaml',
           'ae_kl_aniso_ssim.yaml', 'ae_kl_aniso_structural.yaml', 'ae_kl_aniso_perceptual.yaml',
           'ae_kl_low_compression.yaml', 'ae_kl_aniso_deconv.yaml', 'ldm_baseline.yaml', 'ldm_minsnr.yaml', 'ldm_minsnr_cfg.yaml',
           'ldm_crossattn.yaml', 'ldm_v.yaml')
