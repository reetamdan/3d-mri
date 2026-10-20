import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.models.vqvae import VQVAE3D
from src.models.diffusion import UNet3D, DDPM
from src.data.mrnet import prepare_mrnet
from src.training.common import model_config
from src.training.experiments import train_vq, cache_latents, train_diffusion, generate, evaluate_reconstruction
from src.training.evaluation import multiscale_ssim, segmentation_overlap
from src.training.downstream import compare_augmentation, classification_metrics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mrnet_fixture(tmp_path):
    """Artificial file-layout/label fixture; NOT medical research data."""
    root = tmp_path/'mrnet_fixture'
    for split, exams in [('train', ['0000', '0001']), ('valid', ['0002', '0003'])]:
        for plane in ('axial', 'sagittal', 'coronal'):
            folder = root/split/plane
            folder.mkdir(parents=True)
            for exam in exams:
                rng = np.random.default_rng(int(exam))
                np.save(folder/f'{exam}.npy', rng.uniform(0, 255, (8, 16, 16)).astype(np.float32))
        for task in ('abnormal', 'acl', 'meniscus'):
            (root/f'{split}-{task}.csv').write_text(''.join(f'{exam},{int(exam)%2}\n' for exam in exams))
    return root


def test_mrnet_official_split_and_labels(mrnet_fixture, tmp_path):
    out = tmp_path/'prepared'
    report = prepare_mrnet(mrnet_fixture, out, shape=(16, 16, 16))
    assert report['valid_scans'] == 4
    assert report['split_scans'] == {'train': 2, 'validation': 2, 'test': 0}
    assert json.loads((out/'experiment.json').read_text())['patient_identifiers_available'] is False


def test_vq_gradient_and_shape():
    torch.set_num_threads(2)
    model = VQVAE3D(base=4, latent_channels=4, codes=8)
    x = torch.rand(1, 1, 16, 16, 16)*2-1
    y, loss, perplexity = model(x)
    assert y.shape == x.shape
    ((y-x).square().mean()+loss).backward()
    assert model.quantizer.embedding.weight.grad.abs().sum() > 0
    assert next(model.encoder.parameters()).grad.abs().sum() > 0
    assert 1 <= perplexity <= 8.001


def test_noise_and_seeded_sampling():
    schedule = DDPM(4)
    model = UNet3D(channels=4, base=4, label_dim=3)
    x, noise = torch.randn(1, 4, 4, 4, 4), torch.zeros(1, 4, 4, 4, 4)
    t = torch.tensor([2])
    torch.testing.assert_close(schedule.add_noise(x, t, noise), schedule.cumulative[2].sqrt()*x)
    labels = torch.tensor([[1., 0., 1.]])
    a = schedule.sample(model, x.shape, labels, seed=4)
    b = schedule.sample(model, x.shape, labels, seed=4)
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match='label'):
        model(x, t)


def test_multiscale_and_overlap():
    a = np.random.default_rng(1).uniform(0, 2, (16, 16, 16))
    score, levels = multiscale_ssim(a, a)
    assert score == pytest.approx(1) and levels == 3
    b = np.random.default_rng(2).uniform(0, 2, a.shape)
    assert multiscale_ssim(a, b)[0] < score
    assert segmentation_overlap(a > 1, a > 1) == {'dice': 1., 'iou': 1.}
    y = np.array([[0, 0, 0], [1, 1, 1]])
    assert classification_metrics(y, y)['acl']['auc'] == 1.


def test_all_model_stages_and_augmentation(mrnet_fixture, tmp_path):
    prepared = tmp_path/'prepared'
    prepare_mrnet(mrnet_fixture, prepared, shape=(16, 16, 16))
    c = model_config(ROOT/'configs/smoke-models.yaml')
    c.update(shape=[16, 16, 16], epochs=1, steps_per_epoch=1, conditional=True, diffusion_steps=4,
             select_best=True, initialize_codebook=True, attention=True)
    vq = train_vq(prepared, tmp_path/'vq', c)
    c['epochs'] = 2
    train_vq(prepared, tmp_path/'vq', c, resume=True)
    cache_latents(prepared, vq, tmp_path/'latents', 'cpu')
    diffusion = train_diffusion(tmp_path/'latents', tmp_path/'diffusion', c)
    c['epochs'] = 3
    train_diffusion(tmp_path/'latents', tmp_path/'diffusion', c, resume=True)
    assert (tmp_path/'vq/best.pt').exists() and (tmp_path/'diffusion/best.pt').exists()
    generated = tmp_path/'generated'
    generate(vq, diffusion, generated, count=2, labels=[1, 0, 1], device='cpu')
    generated_manifest = json.loads((generated/'manifest.json').read_text())
    assert (generated/generated_manifest['rows'][0]['latent_file']).exists()
    report = evaluate_reconstruction(prepared, vq, tmp_path/'reconstruction', device='cpu')
    assert len(report['rows']) == 2
    downstream = compare_augmentation(prepared, generated, tmp_path/'downstream',
        shape=(16, 16, 16), limit=2, steps=2, seeds=[42], device='cpu', smoke_test=True)
    assert len(downstream['results']) == 2
    assert {r['arm'] for r in downstream['results']} == {'real_only', 'real_plus_synthetic'}
    with pytest.raises(ValueError, match='limited-data'):
        compare_augmentation(prepared, generated, tmp_path/'bad', shape=(16, 16, 16), limit=1, smoke_test=True)
    with pytest.raises(ValueError, match='labels'):
        generate(vq, diffusion, tmp_path/'missing-labels', device='cpu')
