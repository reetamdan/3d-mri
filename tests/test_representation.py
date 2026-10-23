"""Shape, posterior, structural-gradient, and official-library reference checks."""
import pytest
import torch

from src.models.autoencoder import KLAutoencoder3D, build_autoencoder, autoencoder_step, encode_latent, decode_latent
from src.training.objectives import gradient_loss, ms_ssim_score, ReconstructionObjective, calibrate_weights
from src.training.evaluation import multiscale_ssim


@pytest.mark.parametrize('strides,expected', [([[2, 2, 2], [2, 2, 2]], (4, 4, 4)),
                                          ([[2, 2, 2], [2, 2, 1]], (4, 4, 8)),
                                          ([[2, 2, 2], [1, 1, 1]], (8, 8, 8))])
def test_monai_kl_sampling_adapter_posterior_and_roundtrip(strides, expected, tmp_path):
    pytest.importorskip('monai')
    torch.set_num_threads(2); torch.manual_seed(42)
    model = KLAutoencoder3D(channels=[4, 8, 8], norm_groups=4, latent_channels=2, strides=strides)
    x = torch.randn(1, 1, 16, 16, 16).clamp(-1, 1)
    y, reg = autoencoder_step(model, x, 128)
    assert y.shape == x.shape and torch.isfinite(reg) and reg >= 0
    (F_loss(y, x)+1e-6*reg).backward()
    assert next(model.parameters()).grad.abs().sum() > 0
    model.eval()
    z = encode_latent(model, x)
    assert z.shape == (1, 2, *expected)
    torch.testing.assert_close(decode_latent(model, z), model(x)[0], rtol=0, atol=0)
    torch.testing.assert_close(model(x)[0], model(x)[0], rtol=0, atol=0)
    assert not torch.equal(model(x, sample=True)[0], model(x, sample=True)[0])
    path = tmp_path/'model.pt'; torch.save(model.state_dict(), path)
    loaded = build_autoencoder(model.config).eval(); loaded.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(loaded(x)[0], model(x)[0], rtol=0, atol=0)


def F_loss(a, b):
    return (a-b).abs().mean()


def test_anisotropic_adapter_preserves_initial_parameters():
    pytest.importorskip('monai')
    torch.manual_seed(10); iso = KLAutoencoder3D(channels=[4, 8, 8], norm_groups=4)
    torch.manual_seed(10); aniso = KLAutoencoder3D(channels=[4, 8, 8], norm_groups=4, strides=[[2, 2, 2], [2, 2, 1]])
    torch.testing.assert_close(iso.state_dict(), aniso.state_dict(), rtol=0, atol=0)
    with pytest.raises(ValueError):
        aniso(torch.zeros(1, 1, 15, 16, 16))


def test_learned_upsampling_adapter_has_exact_shape_and_input_gradients():
    pytest.importorskip('monai')
    model = KLAutoencoder3D(channels=[4, 8, 8], norm_groups=4, latent_channels=2,
                            strides=[[2, 2, 2], [2, 2, 1]], use_convtranspose=True)
    x = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
    output = model(x)[0]
    assert output.shape == x.shape
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_structural_objectives_reference_spacing_and_gradients():
    torch.set_num_threads(2); torch.manual_seed(1)
    a = torch.rand(1, 1, 16, 16, 16)*2-1
    b = (a+.05*torch.randn_like(a)).requires_grad_(True)
    measured = ms_ssim_score(b, a)
    reference, levels = multiscale_ssim(b.detach().numpy()[0, 0], a.numpy()[0, 0])
    assert float(measured.detach()) == pytest.approx(reference, abs=1e-6) and levels == 3
    assert float(ms_ssim_score(a, a)) == pytest.approx(1., abs=1e-6)
    assert gradient_loss(a, a) == 0
    assert float(gradient_loss(b, a, (2, 2, 2)).detach()) == pytest.approx(float(gradient_loss(b, a).detach())/2)
    weights = calibrate_weights(b, a)
    loss, terms = ReconstructionObjective(ssim=weights['ssim'], gradient=weights['gradient'])(b, a, b.new_zeros(()))
    loss.backward()
    assert torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0
    assert float(terms['gradient_loss'].detach()) > 0


def test_structural_loss_rejects_missing_features_and_bad_spacing():
    with pytest.raises(ValueError):
        ReconstructionObjective(perceptual=1.)
    with pytest.raises(ValueError):
        gradient_loss(torch.zeros(1, 1, 8, 8, 8), torch.zeros(1, 1, 8, 8, 8), (1, 0, 1))


def test_representation_trainer_resume_and_evaluation(labeled_tensors, tmp_path):
    pytest.importorskip('monai')
    from src.training.representation import train_representation, evaluate_representation
    from src.training.common import read_checkpoint
    config = dict(shape=[16]*3, device='cpu', threads=2, seed=42, epochs=2, batch_size=2,
                  steps_per_epoch=1, learning_rate=.001, gradient_accumulation=1, smoke_test=True,
                  model=dict(family='kl', channels=[4, 8, 8], latent_channels=2, norm_groups=4),
                  loss=dict(regularizer=1e-6))
    full = train_representation(labeled_tensors, tmp_path/'full', config)
    train_representation(labeled_tensors, tmp_path/'resumed', dict(config, epochs=1))
    train_representation(labeled_tensors, tmp_path/'resumed', config, resume=True)
    a, b = read_checkpoint(tmp_path/'full/last.pt'), read_checkpoint(tmp_path/'resumed/last.pt')
    torch.testing.assert_close(a['model'], b['model'], rtol=0, atol=0)
    torch.testing.assert_close(a['optimizer'], b['optimizer'], rtol=0, atol=0)
    result = evaluate_representation(labeled_tensors, full, tmp_path/'evaluation', 'cpu')
    assert len(result['rows']) == 2 and result['latent_shape'] == [2, 4, 4, 4]
    assert result['smoke_test'] and result['means']['gradient_energy_ratio'] >= 0
    with pytest.raises(ValueError, match='same representation config'):
        train_representation(labeled_tensors, tmp_path/'resumed', dict(config, seed=43), resume=True)
    with pytest.raises(ValueError, match='no silent resize'):
        train_representation(labeled_tensors, tmp_path/'wrong-shape', dict(config, shape=[8]*3))


def test_array_grid_audit_reports_missing_physical_metadata(labeled_tensors, tmp_path):
    from scripts.audit_knee_data import audit
    from src.utils.logging import write_json
    write_json(labeled_tensors/'experiment.json', dict(dataset='MRNet', plane='sagittal'))
    report = audit(labeled_tensors, tmp_path/'audit')
    assert report['split_counts'] == {'train': 2, 'validation': 2}
    assert report['physical_orientation_available'] is False and report['spacing_available'] is False


def test_frozen_perceptual_calibration_and_hash_check(tmp_path):
    pytest.importorskip('monai')
    from src.training.representation import _objective
    from src.utils.integrity import sha256_file
    feature = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 3, padding=1), torch.nn.AdaptiveAvgPool3d(2)).eval()
    path = tmp_path/'test-only-features.pt'
    torch.jit.trace(feature, torch.zeros(1, 1, 16, 16, 16)).save(str(path))
    model = KLAutoencoder3D(channels=[4, 8, 8], norm_groups=4, latent_channels=2)
    train = torch.rand(2, 1, 16, 16, 16)*2-1
    config = dict(batch_size=2, loss=dict(regularizer=1e-6, calibrate_perceptual=True,
                  feature_path=str(path), feature_sha256=sha256_file(path)))
    objective, specification = _objective(model, train, config, 'cpu')
    assert specification['perceptual'] > 0
    output, regularizer = autoencoder_step(model, train, 1)
    loss, _ = objective(output, train, regularizer); loss.backward()
    assert next(model.parameters()).grad.abs().sum() > 0
    assert all(p.grad is None for p in objective.feature_model.parameters())
    config['loss']['feature_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='matching SHA'):
        _objective(model, train, config, 'cpu')


def test_cpu_environment_metadata_does_not_initialize_a_gpu(monkeypatch):
    pytest.importorskip('monai')
    from src.training.representation import environment
    def unexpected_gpu_query(*args):
        pytest.fail('CPU run queried/initialized CUDA device properties')
    monkeypatch.setattr(torch.cuda, 'get_device_name', unexpected_gpu_query)
    assert environment('cpu')['hardware'] == 'CPU'
