from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from src.models.vqvae import VQVAE3D
from src.models.diffusion import DDPM, UNet3D
from src.training.common import model_config, read_checkpoint
from src.training.experiments import train_vq, cache_latents, train_diffusion, generate
from src.training.latents import LatentCache
from src.training.diagnostics import tokenizer_diagnostics
from src.training.common import TrainingDataset
from src.training.export import export_inference_pair

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_rng_snapshot_does_not_initialize_cuda(monkeypatch):
    from src.training.state import rng_state
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    def unexpected_cuda_initialization():
        pytest.fail('CPU checkpoint bookkeeping initialized CUDA')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', unexpected_cuda_initialization)
    assert rng_state()['cuda_rng'] is None


def tiny_config():
    c = model_config(ROOT/'configs/smoke-models.yaml')
    c.update(shape=[16]*3, epochs=3, steps_per_epoch=2, conditional=True, diffusion_steps=4,
             channel_mult=[1, 2, 4], label_dropout=.25, ema_decay=.9, prediction_type='v',
             validation_draws=2, initialize_codebook=True, gan_weight=0, lr_schedule='plateau', lr_patience=1)
    c['vq'].update(codes=8, ema_decay=.95, dead_code_threshold=.01)
    return c


def test_ema_codebook_updates_only_in_training_and_keeps_encoder_gradients():
    torch.set_num_threads(2)
    model = VQVAE3D(base=4, latent_channels=4, codes=8, ema_decay=.9)
    x = torch.randn(1, 1, 16, 16, 16)
    before = model.quantizer.embedding.weight.detach().clone()
    y, loss, _ = model(x)
    ((y-x).square().mean()+loss).backward()
    assert not torch.equal(before, model.quantizer.embedding.weight)
    assert next(model.encoder.parameters()).grad.abs().sum() > 0
    assert model.quantizer.embedding.weight.grad is None
    model.eval(); before = model.quantizer.embedding.weight.detach().clone()
    model(x)
    torch.testing.assert_close(before, model.quantizer.embedding.weight, rtol=0, atol=0)


def test_dead_code_reset_uses_training_features():
    from src.models.vqvae import VectorQuantizer
    quantizer = VectorQuantizer(codes=8, channels=4, ema_decay=.99, dead_code_threshold=.5)
    quantizer.ema_counts.zero_(); quantizer.ema_sums.zero_()
    z = torch.randn(1, 4, 3, 3, 3)
    quantizer(z)
    features = z.movedim(1, -1).reshape(-1, 4)
    for embedding in quantizer.embedding.weight:
        assert torch.any(torch.all(embedding == features, dim=1))


def test_multiscale_unet_null_condition_and_odd_shapes():
    model = UNet3D(channels=4, base=4, label_dim=3, channel_mult=[1, 2, 4],
                   attention=True, attention_heads=2, allow_null_labels=True)
    with torch.no_grad():
        model.output.weight.normal_(std=.05)
    x = torch.randn(2, 4, 9, 11, 8)
    t, labels = torch.tensor([0, 3]), torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    conditional = model(x, t, labels)
    null = model(x, t)
    dropped = model(x, t, labels, condition_mask=torch.zeros(2, dtype=torch.bool))
    assert conditional.shape == x.shape and torch.isfinite(conditional).all()
    torch.testing.assert_close(dropped, null)
    assert not torch.allclose(conditional, null)
    conditional.square().mean().backward()
    assert model.middle.conv1.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='too small'):
        model(torch.zeros(1, 4, 2, 2, 2), torch.tensor([0]), labels[:1])


def test_v_prediction_recovers_known_clean_and_noise_at_all_timesteps():
    schedule = DDPM(8, prediction_type='v')
    clean, noise = torch.randn(1, 2, 4, 4, 4), torch.randn(1, 2, 4, 4, 4)
    for step in range(8):
        t = torch.tensor([step])
        noisy = schedule.add_noise(clean, t, noise)
        target = schedule.training_target(clean, t, noise)
        restored, recovered_noise = schedule.clean_and_noise(noisy, step, target)
        torch.testing.assert_close(restored, clean, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(recovered_noise, noise, atol=1e-6, rtol=1e-6)


def test_ddim_matches_perfect_zero_clean_oracle_and_checks_guidance():
    schedule = DDPM(8)
    class Oracle(torch.nn.Module):
        config = {'allow_null_labels': False}
        def forward(self, x, t, labels=None):
            return x/(1-schedule.cumulative[t])[:, None, None, None, None].sqrt()
    output = schedule.sample(Oracle(), (1, 2, 4, 4, 4), method='ddim', sampling_steps=3)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-5, rtol=0)
    with pytest.raises(ValueError, match='every step'):
        schedule.sample(Oracle(), output.shape, sampling_steps=2)
    with pytest.raises(ValueError, match='label dropout'):
        schedule.sample(Oracle(), output.shape, labels=torch.ones(1, 3), guidance_scale=2.)


def test_resume_equivalence_packed_latents_and_ema_sampling(labeled_tensors, tmp_path):
    c = tiny_config()
    full = train_vq(labeled_tensors, tmp_path/'vq-full', deepcopy(c))
    partial = deepcopy(c); partial['epochs'] = 1
    train_vq(labeled_tensors, tmp_path/'vq-resume', partial)
    resumed = train_vq(labeled_tensors, tmp_path/'vq-resume', deepcopy(c), resume=True)
    a, b = read_checkpoint(full), read_checkpoint(resumed)
    torch.testing.assert_close(a['model'], b['model'], rtol=0, atol=0)
    torch.testing.assert_close(a['optimizer'], b['optimizer'], rtol=0, atol=0)
    cache_latents(labeled_tensors, full, tmp_path/'latents', device='cpu')
    manifest = json.loads((tmp_path/'latents/manifest.json').read_text())
    cache = LatentCache(tmp_path/'latents', manifest)
    for row in manifest['rows']:
        torch.testing.assert_close(cache.tensor(row), torch.load(tmp_path/'latents'/row['file'], weights_only=True), rtol=0, atol=0)
    diffusion = train_diffusion(tmp_path/'latents', tmp_path/'diff-full', deepcopy(c))
    partial = deepcopy(c); partial['epochs'] = 1
    train_diffusion(tmp_path/'latents', tmp_path/'diff-resume', partial)
    resumed = train_diffusion(tmp_path/'latents', tmp_path/'diff-resume', deepcopy(c), resume=True)
    a, b = read_checkpoint(diffusion), read_checkpoint(resumed)
    for key in ('model', 'optimizer', 'ema', 'ema_model'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    generate(full, diffusion, tmp_path/'samples', count=2, labels=[1, 1, 1], device='cpu',
             sampler='ddim', sampling_steps=2, guidance_scale=1.5)
    generated = json.loads((tmp_path/'samples/manifest.json').read_text())
    assert generated['sampling']['weights'] == 'ema'
    assert generated['sampling']['prediction_type'] == 'v'
    assert len(generated['rows'][0]['sampling_diagnostics']) == 2
    exported = export_inference_pair(full, diffusion, tmp_path/'compact')
    assert exported['exported_bytes'] < exported['original_bytes']
    generate(tmp_path/'compact/vq.pt', tmp_path/'compact/diffusion.pt', tmp_path/'compact-samples', count=2,
             labels=[1, 1, 1], device='cpu', sampler='ddim', sampling_steps=2, guidance_scale=1.5)
    compact = json.loads((tmp_path/'compact-samples/manifest.json').read_text())
    assert compact['rows'][0]['sha256'] == generated['rows'][0]['sha256']
    assert read_checkpoint(tmp_path/'compact/diffusion.pt')['inference_only']
    model = VQVAE3D(**read_checkpoint(full)['model_config']); model.load_state_dict(read_checkpoint(full)['model'])
    diagnostics = tokenizer_diagnostics(model, TrainingDataset(labeled_tensors, 'train', c['shape']), 'cpu')
    assert sum(diagnostics['counts']) == 2*4**3
    assert 1 <= diagnostics['used_codes'] <= 8
    cache.close()
    packed = tmp_path/'latents'/manifest['packed']['train']['file']
    array = np.load(packed); array.flat[0] += 1; np.save(packed, array)
    with pytest.raises(ValueError, match='checksum'):
        LatentCache(tmp_path/'latents', manifest)


def test_plateau_scheduler_and_early_stopping_use_validation(labeled_tensors, tmp_path, monkeypatch):
    import src.training.experiments as experiments
    original = experiments.F.l1_loss
    def fixed_validation(a, b):
        return original(a, b) if torch.is_grad_enabled() else a.new_tensor(.5)
    monkeypatch.setattr(experiments.F, 'l1_loss', fixed_validation)
    c = tiny_config()
    c.update(epochs=10, early_stopping_patience=1, lr_patience=0)
    checkpoint = train_vq(labeled_tensors, tmp_path/'early-stop', c)
    saved = read_checkpoint(checkpoint)
    assert saved['epoch'] == 2
    assert saved['optimizer']['param_groups'][0]['lr'] == c['learning_rate']*.5


def test_gradient_accumulation_counts_optimizer_updates(labeled_tensors, tmp_path):
    c = tiny_config(); c.update(epochs=1, gradient_accumulation=2)
    vq = train_vq(labeled_tensors, tmp_path/'accumulated-vq', c)
    history = read_checkpoint(vq)['history'][0]
    assert history['micro_batches'] == 2 and history['optimizer_steps'] == 1
    cache_latents(labeled_tensors, vq, tmp_path/'accumulated-cache', 'cpu')
    diffusion = train_diffusion(tmp_path/'accumulated-cache', tmp_path/'accumulated-diffusion', c)
    history = read_checkpoint(diffusion)['history'][0]
    assert history['micro_batches'] == 4 and history['optimizer_steps'] == 2
