"""Official-formula Min-SNR checks and complete small KL-latent integration."""
import json
import pytest
import torch
from torch.nn import functional as F

from src.models.diffusion import DDPM, UNet3D, min_snr_weights, append_coordinates


def test_min_snr_matches_reference_target_specific_weights():
    snr = torch.tensor([0., 1., 5., 10.])
    torch.testing.assert_close(min_snr_weights(snr, 5, 'epsilon'), torch.tensor([1., 1., 1., .5]))
    torch.testing.assert_close(min_snr_weights(snr, 5, 'v'), torch.tensor([0., .5, 5/6, 5/11]))
    torch.testing.assert_close(min_snr_weights(snr, 5, 'x0'), torch.tensor([0., 1., 5., 5.]))
    with pytest.raises(ValueError):
        min_snr_weights(snr, -1)


def test_coordinate_channels_are_array_indices_and_preserve_prediction_shape():
    torch.set_num_threads(2)
    x = torch.zeros(1, 2, 4, 6, 8, requires_grad=True)
    coordinates = append_coordinates(x)
    assert coordinates.shape == (1, 5, 4, 6, 8)
    assert coordinates[0, 2, 0, 2, 2] == -1 and coordinates[0, 2, -1, 2, 2] == 1
    assert coordinates[0, 4, 2, 2, 0] == -1 and coordinates[0, 4, 2, 2, -1] == 1
    model = UNet3D(channels=2, base=4, coordinate_channels=True)
    output = model(x, torch.tensor([1])); assert output.shape == x.shape
    output.square().mean().backward(); assert torch.isfinite(x.grad).all()
    torch.manual_seed(12); baseline = UNet3D(channels=2, base=4)
    state_after_baseline = torch.get_rng_state()
    torch.manual_seed(12); matched = UNet3D(channels=2, base=4, coordinate_channels=True)
    torch.testing.assert_close(torch.get_rng_state(), state_after_baseline, rtol=0, atol=0)
    torch.testing.assert_close(matched(x, torch.tensor([1])), baseline(x, torch.tensor([1])), rtol=0, atol=0)


@pytest.mark.parametrize('channels', [1, 4])
def test_coordinate_initialization_matches_identity_and_projection_skips(channels):
    torch.manual_seed(8); baseline = UNet3D(channels=channels, base=4)
    torch.manual_seed(8); coordinate = UNet3D(channels=channels, base=4, coordinate_channels=True)
    x = torch.randn(1, channels, 4, 4, 4)
    torch.testing.assert_close(coordinate(x, torch.tensor([1])), baseline(x, torch.tensor([1])), rtol=0, atol=0)


def test_future_min_snr_ablation_changes_only_the_weighting():
    from src.training.common import model_config
    baseline = model_config(preset='ldm_baseline.yaml', stage='diffusion')
    weighted = model_config(preset='ldm_minsnr.yaml', stage='diffusion')
    assert {key for key in baseline if baseline[key] != weighted[key]} == {'min_snr_gamma'}
    assert weighted['min_snr_gamma'] == 5.


@pytest.mark.parametrize('target_type', ['epsilon', 'v'])
def test_min_snr_weights_each_volume_before_batch_reduction(target_type):
    schedule = DDPM(8, target_type)
    prediction = torch.zeros(2, 3, 4, 4, 4, requires_grad=True)
    target = torch.ones_like(prediction); target[1] *= 2
    t = torch.tensor([0, 7])
    a = schedule.cumulative[t]
    weights = min_snr_weights(a/(1-a), 5, target_type)
    expected = (torch.tensor([1., 4.])*weights).mean()
    loss = schedule.denoising_loss(prediction, target, t, 5)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(schedule.denoising_loss(prediction, target, t), F.mse_loss(prediction, target), rtol=0, atol=0)
    loss.backward(); assert torch.isfinite(prediction.grad).all()


def test_cross_attention_null_dropout_shape_and_gradient():
    torch.manual_seed(42); torch.set_num_threads(2)
    model = UNet3D(channels=2, base=4, label_dim=3, channel_mult=[1, 2], allow_null_labels=True,
                   conditioning_mode='cross_attention', attention_heads=2)
    with torch.no_grad():
        model.output.weight.normal_(std=.03)
        model.context_attention.output.weight.normal_(std=.03)
    x = torch.randn(2, 2, 4, 4, 4)
    t = torch.tensor([1, 2]); labels = torch.tensor([[0., 0., 0.], [1., 1., 0.]])
    first = model(x, t, labels)
    dropped = model(x, t, labels, condition_mask=torch.zeros(2, dtype=torch.bool))
    unconditional = model(x, t)
    assert first.shape == x.shape and not torch.allclose(first, unconditional)
    torch.testing.assert_close(dropped, unconditional, rtol=0, atol=0)
    first.square().mean().backward()
    assert model.condition_tokens.weight.grad.abs().sum() > 0
    copied = UNet3D(**model.config); copied.load_state_dict(model.state_dict())
    torch.testing.assert_close(copied(x, t, labels), first, rtol=0, atol=0)


def test_kl_cached_latents_train_and_sample_with_min_snr(labeled_tensors, tmp_path):
    pytest.importorskip('monai')
    from src.training.representation import train_representation
    from src.training.latent_statistics import cache_representation
    from src.training.experiments import train_diffusion, generate
    c = dict(shape=[16]*3, device='cpu', threads=2, seed=42, epochs=1, batch_size=2,
        steps_per_epoch=1, learning_rate=.001, gradient_accumulation=1, smoke_test=True,
        model=dict(family='kl', channels=[4, 8, 8], latent_channels=2, norm_groups=4), loss=dict(regularizer=1e-6))
    ae = train_representation(labeled_tensors, tmp_path/'ae', c)
    cached = cache_representation(labeled_tensors, ae, tmp_path/'latents', 'cpu')
    d = dict(c, conditional=True, label_dropout=.1, unet_base=4, diffusion_steps=4,
             prediction_type='v', min_snr_gamma=5., channel_mult=[1, 2], ema_decay=.9)
    cached['smoke_test'] = False  # Artificial fixture exercises the research gate, not a real result.
    from src.utils.logging import write_json
    write_json(tmp_path/'latents/manifest.json', cached)
    with pytest.raises(ValueError, match='passed representation gate'):
        train_diffusion(tmp_path/'latents', tmp_path/'blocked', dict(d, smoke_test=False))
    diffusion = train_diffusion(tmp_path/'latents', tmp_path/'diffusion', d)
    generate(ae, diffusion, tmp_path/'generated', count=2, labels=[1, 0, 0], device='cpu', sampler='ddim', sampling_steps=2, guidance_scale=1.5)
    manifest = json.loads((tmp_path/'generated/manifest.json').read_text())
    assert len(manifest['rows']) == 2
    assert manifest['rows'][0]['quantization_mse'] is None
    assert manifest['sampling']['weights'] == 'ema'
    from src.training.conditional_generation import generate_balanced
    generate_balanced(ae, diffusion, tmp_path/'balanced', count=4, seed=42, device='cpu', sampler='ddim', sampling_steps=2)
    balanced = json.loads((tmp_path/'balanced/manifest.json').read_text())
    assert len(balanced['rows']) == 4
    assert {tuple(r['conditioning_labels']) for r in balanced['rows']} == {(0, 0, 0), (1, 1, 1)}
    from src.utils.integrity import generated_volume
    assert generated_volume(tmp_path/'balanced', balanced['rows'][0], balanced['shape']).shape == (1, 16, 16, 16)
