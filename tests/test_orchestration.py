"""CPU-only experiment orchestration on artificial arrays, never research evidence."""
from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from scripts import run_m4, research_study, research
from src.data.mrnet import TASKS
from src.training.common import model_config
from src.utils.integrity import sha256_file
from src.utils.logging import write_json


@pytest.fixture
def experiment_inputs(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    files = []
    for split, offset in (('train', 0), ('valid', 100)):
        folder = source/split/'sagittal'; folder.mkdir(parents=True)
        exams = [f'{offset+i:04d}' for i in range(4)]
        for task in TASKS:
            (source/f'{split}-{task}.csv').write_text(''.join(f'{exam},{int(task == "abnormal")}\n' for exam in exams))
        for i, exam in enumerate(exams):
            path = folder/f'{exam}.npy'
            np.save(path, np.random.default_rng(offset+i).integers(0, 256, (8, 16, 16), dtype=np.uint8))
            files.append(dict(file=path.relative_to(source).as_posix(), sha256=sha256_file(path)))
    write_json(source/'source_manifest.json', dict(complete=True, files=files, selection={'version': 1},
        source_url='fixture://artificial-arrays', upstream_url='fixture://none', mirror_license='Numerical fixture',
        class_counts={}, source_array_bytes=sum((source/r['file']).stat().st_size for r in files),
        split_policy='Artificial software-test identities'))
    config = model_config(preset='smoke-models.yaml')
    config.update(shape=[16]*3, epochs=1, steps_per_epoch=1, batch_size=2, conditional=True,
                  smoke_test=False, select_best=True, gan_weight=0, diffusion_steps=4)
    path = tmp_path/'models.yaml'; path.write_text(yaml.safe_dump(config))
    return source, path


def test_m4_runner_stage_sequence_resume_and_cache_lineage(experiment_inputs, tmp_path, monkeypatch):
    source, config = experiment_inputs
    monkeypatch.setattr(run_m4.torch.cuda, 'is_available', lambda: False)
    run = tmp_path/'m4'
    args = Namespace(source=source, data=tmp_path/'processed', output=run, vq_config=config,
                     diffusion_config=config, count=2, resume=False)
    run_m4.run(args)
    status = json.loads((run/'status.json').read_text())
    assert list(status['stages']) == ['preprocess', 'train_vq', 'cache_latents', 'train_diffusion',
                                      'held_out_reconstruction', 'generation', 'generation_diagnostics', 'evidence']
    assert status['current_stage'] == 'complete'
    checkpoint = run/'vq/best.pt'; original = sha256_file(checkpoint)
    manifest = sha256_file(run/'generated/manifest.json')
    with pytest.raises(ValueError, match='Experiment exists'):
        run_m4.run(args)
    args.resume = True; run_m4.run(args)
    assert sha256_file(checkpoint) == original and sha256_file(run/'generated/manifest.json') == manifest
    status = json.loads((run/'status.json').read_text())
    assert len(status['previous_sessions']) == 1 and 'preprocess' not in status['stages']
    cache = run/'latents/manifest.json'; content = json.loads(cache.read_text()); content['vq_sha256'] = 'wrong'
    write_json(cache, content)
    with pytest.raises(ValueError, match='cache no longer matches'):
        run_m4.run(args)


def test_study_runner_supports_cpu_presets_and_reuses_completed_arms(experiment_inputs, tmp_path, monkeypatch):
    source, config = experiment_inputs
    monkeypatch.setattr(research_study.torch.cuda, 'is_available', lambda: False)
    run = tmp_path/'study'
    args = Namespace(source=source, data=tmp_path/'processed', output=run, baseline_config=config,
                     upgraded_config=config, seeds=[42], vq_epochs=1, diffusion_epochs=1, steps=1, count=2, resume=False)
    research_study.run(args)
    report = json.loads((run/'study.json').read_text())
    assert len(report['records']) == 2 and not report['independent_test']
    assert {r['arm'] for r in report['records']} == {'baseline', 'upgraded'}
    original = {name: sha256_file(run/name/'vq/best.pt') for name in ('baseline-42', 'upgraded-42')}
    args.resume = True; research_study.run(args)
    assert all(sha256_file(run/name/'vq/best.pt') == digest for name, digest in original.items())
    args.steps = 2
    with pytest.raises(ValueError, match='same predeclared plan'):
        research_study.run(args)
    args.steps = 1
    (run/'upgraded-42/vq/best.pt').write_bytes(b'altered artifact')
    with pytest.raises(ValueError, match='checkpoint changed'):
        research_study.run(args)


def test_nifti_script_entrypoints_share_configuration_and_export(config, nifti, tmp_path, monkeypatch):
    import shutil
    from scripts import preprocess, validate_dataset, generate_report
    folder = Path(config['dataset']['raw_dir'])/'fixture-patient'; folder.mkdir()
    shutil.copyfile(nifti, folder/'scan.nii.gz')
    path = tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(config))
    for module in (validate_dataset, generate_report, preprocess):
        monkeypatch.setattr(sys, 'argv', ['script', '--config', str(path)])
        module.main()
    metadata = Path(config['dataset']['metadata_dir'])
    assert json.loads((metadata/'quality_report.json').read_text())['valid_scans'] == 1
    assert json.loads((metadata/'latest_run.json').read_text())['report']['valid_scans'] == 1


@pytest.mark.parametrize('command', ['train-vq', 'train-diffusion'])
def test_training_cli_default_preset_is_independent_of_cwd(command, tmp_path, monkeypatch, capsys):
    from src.training import experiments
    monkeypatch.chdir(tmp_path)
    received = {}
    def train(data, output, config, resume):
        received.update(data=data, output=output, config=config, resume=resume)
        return output
    monkeypatch.setattr(experiments, 'train_vq' if command == 'train-vq' else 'train_diffusion', train)
    monkeypatch.setattr(sys, 'argv', ['mri3d', command, '--data', 'fixture', '--output', 'result', '--resume'])
    research.main()
    assert received['config']['threads'] == 4 and received['resume']
    assert received['data'] == 'fixture' and json.loads(capsys.readouterr().out) == 'result'
