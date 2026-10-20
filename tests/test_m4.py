"""Behavioral checks for the real-data workflow additions (offline fixtures only)."""
import json
import numpy as np
import pytest
import torch
from scripts.download_mrnet_subset import stratified_ids, download_subset, fetch
from src.models.diffusion import UNet3D, SpatialAttention3D, DDPM
from src.training.evaluation import evaluate_generation
from src.utils.integrity import sha256_file
from src.training.common import fingerprint


def test_stratified_selection_is_reproducible_and_uses_real_ids():
    labels = {f'{i:04d}': [i % 2, i % 3 == 0, i % 5 == 0] for i in range(100)}
    a = stratified_ids(labels, 24, 42)
    assert len(a) == len(set(a)) == 24
    assert set(a).issubset(labels)
    assert a == stratified_ids(labels, 24, 42)
    assert a != stratified_ids(labels, 24, 43)
    with pytest.raises(ValueError):
        stratified_ids(labels, 101, 42)


def test_attention_forward_backward_and_configuration():
    torch.set_num_threads(2)
    model = UNet3D(channels=4, base=4, label_dim=3, attention=True, attention_heads=2)
    x = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
    y = model(x, torch.tensor([1, 2]), torch.tensor([[0., 0., 0.], [1., 0., 1.]]))
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert model.attention.project.weight.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0
    assert isinstance(UNet3D(**model.config).attention, SpatialAttention3D)
    with pytest.raises(ValueError, match='divide'):
        SpatialAttention3D(8, heads=3)


def test_denoised_bounds_prevent_untrained_sampler_explosion():
    class ZeroNoise(torch.nn.Module):
        def forward(self, x, t, labels):
            return torch.zeros_like(x)
    schedule = DDPM(8)
    result = schedule.sample(ZeroNoise(), (1, 2, 4, 4, 4), seed=3, clip_range=(-2., 3.))
    assert torch.isfinite(result).all() and result.min() >= -2.0001 and result.max() <= 3.0001
    torch.testing.assert_close(result, schedule.sample(ZeroNoise(), result.shape, seed=3, clip_range=(-2., 3.)))


def test_kaggle_single_file_archive_is_decoded(monkeypatch):
    import io
    import zipfile
    import scripts.download_mrnet_subset as downloader
    payload = b'expected-volume-bytes'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('0000.npy', payload)
    class Response(io.BytesIO):
        pass
    monkeypatch.setattr(downloader, 'urlopen', lambda *args, **kwargs: Response(buffer.getvalue()))
    assert fetch('train/sagittal/0000.npy') == payload


def test_download_budget_is_enforced_before_writing_arrays(tmp_path, monkeypatch):
    import io
    import scripts.download_mrnet_subset as downloader
    volume = np.arange(3*256*256, dtype=np.uint8).reshape(3, 256, 256)
    buffer = io.BytesIO(); np.save(buffer, volume)
    def source(relative):
        if relative.endswith('.csv'):
            return b'1130,0\n1131,1\n' if relative.startswith('valid') else b'0000,0\n0001,1\n'
        return buffer.getvalue()
    monkeypatch.setattr(downloader, 'fetch', source)
    with pytest.raises(ValueError, match='byte budget'):
        download_subset(tmp_path/'subset', 2, 2, max_mib=1, workers=2)
    assert sum(p.stat().st_size for p in (tmp_path/'subset').rglob('*.npy')) <= 1024**2


def test_generation_evaluation_checks_lineage_and_reports_real_computations(tmp_path):
    import csv
    real, generated = tmp_path/'real', tmp_path/'generated'
    real.mkdir(); generated.mkdir()
    rng = np.random.default_rng(2)
    volumes = [rng.uniform(-1, 1, (1, 8, 8, 8)).astype(np.float32) for _ in range(2)]
    rows = []
    for i, volume in enumerate(volumes):
        np.save(real/f'{i}.npy', volume)
        rows.append(dict(scan_id=str(i), patient_id=f'p{i}', split='train', output_file=f'{i}.npy',
                         output_sha256=sha256_file(real/f'{i}.npy')))
    with (real/'metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    samples = []
    for i, volume in enumerate(volumes):
        np.save(generated/f'{i}.npy', volume)
        samples.append(dict(file=f'{i}.npy', sha256=sha256_file(generated/f'{i}.npy')))
    manifest = dict(rows=samples, shape=[8, 8, 8], generator_train_ids=['0', '1'],
                    smoke_test=True, dataset_fingerprint=fingerprint(real))
    (generated/'manifest.json').write_text(json.dumps(manifest))
    report = evaluate_generation(generated, real, tmp_path/'report.json', max_samples=2)
    assert report['generated_count'] == report['real_count'] == 2
    assert report['generated_diversity']['pairs'] == 1
    assert report['nearest_training_mse'] == [0., 0.]
    with pytest.raises(ValueError, match='At least two'):
        evaluate_generation(generated, real, tmp_path/'bad.json', max_samples=1)
    manifest['dataset_fingerprint'] = 'wrong'
    (generated/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='provenance'):
        evaluate_generation(generated, real, tmp_path/'bad.json')
