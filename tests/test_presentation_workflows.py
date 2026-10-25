"""Small artificial raw/model layouts verify UI orchestration, never medical results."""
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.utils.integrity import sha256_file
from src.utils.logging import write_json


def dataset_view(root):
    from pathlib import Path
    from app.dataset_explorer import dataset_page
    dataset_page(Path(root))


def generation_view(root):
    from pathlib import Path
    from app.generation_lab import generation_page
    generation_page(Path(root))


def test_acquired_dataset_controls_integrity_and_isolated_preparation(tmp_path):
    root = tmp_path/'workspace'; source = root/'data/mrnet/MRNet-v1.0'
    files = []
    for split, ids in (('train', ['0000', '0001']), ('valid', ['0002', '0003'])):
        for plane in ('axial', 'sagittal', 'coronal'):
            folder = source/split/plane; folder.mkdir(parents=True)
            for identity in ids:
                path = folder/f'{identity}.npy'
                np.save(path, np.random.default_rng(int(identity)).uniform(0, 255, (8, 16, 16)).astype(np.float32))
                files.append(dict(file=path.relative_to(source).as_posix(), sha256=sha256_file(path)))
        for task in ('abnormal', 'acl', 'meniscus'):
            (source/f'{split}-{task}.csv').write_text(''.join(f'{identity},{int(identity)%2}\n' for identity in ids))
    write_json(source/'source_manifest.json', dict(files=files, purpose='Artificial UI layout fixture'))
    app = AppTest.from_function(dataset_view, args=(str(root),)).run(timeout=30)
    assert not app.exception and not app.error and len(app.slider) == 3
    next(s for s in app.selectbox if s.label == 'Official split').select('valid').run()
    next(s for s in app.selectbox if s.label == 'Exam').select_index(1).run()
    for slider in app.slider:
        slider.set_value(slider.max)
    app.run(); assert not app.exception
    next(t for t in app.text_input if t.label == 'New processed run folder').set_value(str(tmp_path/'prepared'))
    next(n for n in app.number_input if n.label == 'Maximum exams per official split').set_value(1)
    for number in app.number_input:
        if number.label.startswith('Target '):
            number.set_value(16)
    next(b for b in app.button if b.label == 'Validate & prepare MRNet subset').click().run(timeout=30)
    assert not app.exception and json.loads((tmp_path/'prepared/report.json').read_text())['valid_scans'] == 2
    changed = source/'valid/axial/0003.npy'; changed.write_bytes(b'corrupt acquired fixture')
    app.run()
    assert not app.exception and any('could not be read' in e.value for e in app.error)


def generator_fixture(root):
    run = root/'runs/ARTIFICIAL-MODEL-UI'
    hashes = {}
    for stage in ('vq', 'diffusion'):
        folder = run/stage; folder.mkdir(parents=True)
        checkpoint = folder/'last.pt'; checkpoint.write_bytes(f'Not a model: UI-only fixture {stage}'.encode())
        hashes[f'{stage}_sha256'] = sha256_file(checkpoint)
        write_json(folder/'run.json', dict(smoke_test=False, completed_epochs=1, checkpoint_file='last.pt', checkpoint_sha256=sha256_file(checkpoint)))
        write_json(folder/'history.json', [dict(epoch=1, reconstruction_l1=.1, noise_mse=.2)])
    folder = run/'generated'; folder.mkdir()
    array = folder/'fixture.npy'; np.save(array, np.zeros((1, 8, 8, 8), dtype=np.float32))
    manifest = dict(shape=[8]*3, smoke_test=False, **hashes,
        sampling=dict(training_steps=4, classifier_free_guidance_trained=False),
        rows=[dict(file=array.name, sha256=sha256_file(array), seed=42, conditioning_labels=[1, 0, 0])])
    write_json(folder/'manifest.json', manifest)
    return run, manifest


def test_generation_ui_passes_requested_seed_sampler_and_labels_to_fresh_bound_output(tmp_path, monkeypatch):
    import src.training.experiments as experiments
    run, manifest = generator_fixture(tmp_path)
    calls = []
    def generate(vq, diffusion, output, *, count, seed, labels, sampler, sampling_steps, guidance_scale):
        calls.append(dict(seed=seed, labels=labels, sampler=sampler, steps=sampling_steps))
        output.mkdir(parents=True)
        path = output/'fixture.npy'; np.save(path, np.full((1, 8, 8, 8), .25, dtype=np.float32))
        write_json(output/'manifest.json', dict(manifest, rows=[dict(file=path.name, sha256=sha256_file(path), seed=seed, conditioning_labels=labels)]))
    monkeypatch.setattr(experiments, 'generate', generate)
    app = AppTest.from_function(generation_view, args=(str(tmp_path),)).run(timeout=30)
    assert not app.exception and not app.error
    next(n for n in app.number_input if n.label == 'Sampling seed').set_value(7)
    next(b for b in app.button if b.label == 'Generate one volume').click().run(timeout=30)
    assert not app.exception and calls == [dict(seed=7, labels=[1, 0, 0], sampler='ddim', steps=4)]
    latest = app.session_state['generation_latest']
    assert Path(latest['folder']).is_relative_to(run/'interactive')
    assert any('Latest live request' in s.options[0] for s in app.selectbox if s.label == 'Diffusion sample')


def test_generation_view_rejects_changed_checkpoint_and_bad_stage_flags(tmp_path):
    from app.generation_lab import generator_artifacts
    run, _ = generator_fixture(tmp_path)
    generator_artifacts(run)
    (run/'diffusion/last.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash-verified'):
        generator_artifacts(run)
