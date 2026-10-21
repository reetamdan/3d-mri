"""Hardware-path checks; CPU CI skips these, the manual GPU job requires CUDA."""
from copy import deepcopy
import pytest
import torch
from src.models.vqvae import VQVAE3D
from src.models.diffusion import UNet3D, DDPM
from src.training.state import ModelEMA, precision

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')


def test_cuda_mixed_precision_multiscale_ema_v_prediction_and_ddim():
    device = torch.device('cuda')
    model = UNet3D(channels=4, base=8, label_dim=3, channel_mult=[1, 2, 4],
                   attention=True, allow_null_labels=True).to(device)
    ema = ModelEMA(model, .99)
    amp, dtype, scaler = precision({'mixed_precision': True}, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    z = torch.randn(2, 4, 8, 8, 8, device=device)
    labels, timesteps = torch.ones(2, 3, device=device), torch.tensor([1, 3], device=device)
    schedule = DDPM(8, prediction_type='v').to(device)
    noise = torch.randn_like(z)
    with torch.autocast(device_type='cuda', dtype=dtype, enabled=amp):
        prediction = model(schedule.add_noise(z, timesteps, noise), timesteps, labels,
                           condition_mask=torch.tensor([True, False], device=device))
        loss = (prediction-schedule.training_target(z, timesteps, noise)).square().mean()
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); ema.update(model)
    evaluation = deepcopy(model).eval(); evaluation.load_state_dict(ema.shadow)
    sample = schedule.sample(evaluation, (1, 4, 8, 8, 8), labels[:1], seed=5,
                             method='ddim', sampling_steps=4, guidance_scale=1.5, clip_range=(-3., 3.))
    assert torch.isfinite(sample).all() and ema.updates == 1
    vq = VQVAE3D(base=4, latent_channels=4, codes=8, ema_decay=.9).to(device)
    output, loss, _ = vq(torch.randn(1, 1, 16, 16, 16, device=device))
    (output.square().mean()+loss).backward()
    assert torch.isfinite(output).all()
