"""End-to-end stage-one plan/gate reuse and diagnostic interpretation on small fixtures."""
from argparse import Namespace
import json
import pytest
import torch
import yaml

from src.training.common import model_config
from src.training.experiments import train_vq
from src.utils.integrity import sha256_file


def test_representation_plan_profile_compare_and_resume(labeled_tensors, tmp_path):
    pytest.importorskip('monai')
    from scripts.representation_study import run
    c = model_config(preset='smoke-models.yaml'); c.update(shape=[16]*3, epochs=1, steps_per_epoch=1, gan_weight=0)
    baseline = train_vq(labeled_tensors, tmp_path/'baseline', c)
    configs = tmp_path/'configs'; configs.mkdir()
    common = dict(shape=[16]*3, device='cpu', threads=2, seed=42, epochs=1, batch_size=2,
                  steps_per_epoch=1, learning_rate=.001, gradient_accumulation=1, smoke_test=True)
    vq = dict(common, model=dict(family='vq', base=4, latent_channels=4, codes=16, commitment=.25), loss=dict(regularizer=1.))
    kl = dict(common, model=dict(family='kl', channels=[4, 8, 8], latent_channels=2, norm_groups=4,
                                strides=[[2, 2, 2], [2, 2, 1]]), loss=dict(regularizer=1e-6))
    for name, value in (('ae_vq_baseline', vq), ('ae_kl_aniso', kl)):
        (configs/f'{name}.yaml').write_text(yaml.safe_dump(value))
    args = Namespace(data=labeled_tensors, output=tmp_path/'study', config_dir=str(configs),
        arms=['ae_vq_baseline', 'ae_kl_aniso'], seeds=[42], baseline=baseline, device='cpu',
        profile_only=True, profile_updates=1, resume=False, feature_extractor=None)
    run(args)
    assert len(json.loads((args.output/'profiles.json').read_text())) == 2
    args.profile_only = False; args.resume = True; run(args)
    study = json.loads((args.output/'study.json').read_text())
    assert study['status'] == 'completed' and study['smoke_test']
    assert len(study['records']) == 2 and (args.output/'comparison-1.png').is_file()
    checkpoint = args.output/'ae_kl_aniso-42/training/best.pt'; before = sha256_file(checkpoint)
    run(args); assert sha256_file(checkpoint) == before
    args.seeds = [43]
    with pytest.raises(ValueError, match='plan changed'):
        run(args)


def test_detail_gate_does_not_accept_psnr_alone():
    from scripts.representation_study import gate_result
    frozen = dict(psnr=24., gradient_energy_ratio=.3, edge_mae=.1)
    interpolation = dict(ssim=.77, ms_ssim=.9)
    measured = dict(psnr=28., ssim=.9, ms_ssim=.98, gradient_energy_ratio=.8, edge_mae=.05,
                    high_frequency_energy_ratio=.8)
    assert all(gate_result(measured, frozen, interpolation).values())
    measured['high_frequency_energy_ratio'] = .1
    assert gate_result(measured, frozen, interpolation)['high_frequency_range'] is False


def test_artificial_conditioning_control_does_not_claim_success_when_untrained(tmp_path):
    from scripts.diffusion_sanity import conditioning_control
    torch.set_num_threads(2)
    result = conditioning_control(tmp_path, torch.device('cpu'), updates=1)
    assert result['artificial_fixture'] and not result['passed']
    assert len(result['rows']) == 8
