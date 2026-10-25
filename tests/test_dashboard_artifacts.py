"""Persistent research browsing with isolated numerical/artifact fixtures."""
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.research_workspace import interactive_samples
from src.utils.integrity import sha256_file
from src.utils.logging import write_json
from src.visualization.mri_viewer import TENSOR_SECTIONS, tensor_slice_figure


def workspace(root):
    from pathlib import Path
    from app.research_workspace import experiments_page
    experiments_page(Path(root))


def research_fixture(root, name, shape):
    run = root/'runs'/name
    hashes = {}
    for stage in ('vq', 'diffusion'):
        folder = run/stage; folder.mkdir(parents=True)
        path = folder/'last.pt'; path.write_bytes(f'non-model test artifact {name} {stage}'.encode())
        hashes[f'{stage}_sha256'] = sha256_file(path)
        write_json(folder/'run.json', dict(smoke_test=False, completed_epochs=1,
                   checkpoint_file='last.pt', checkpoint_sha256=sha256_file(path)))
    manifest = dict(shape=list(shape), smoke_test=False, **hashes)
    for folder, seed in ((run/'generated', 0), (run/'interactive/sample-one', 1), (run/'interactive/sample-two', 2)):
        folder.mkdir(parents=True)
        path = folder/'sample.npy'
        np.save(path, np.random.default_rng(seed).uniform(-1, 1, (1, *shape)).astype(np.float32))
        write_json(folder/'manifest.json', dict(manifest, rows=[dict(file=path.name, sha256=sha256_file(path),
                   seed=seed, conditioning_labels=None)]))
    return run, manifest


def test_tensor_sections_follow_array_axes_without_patient_space_labels():
    array = np.arange(8*16*24).reshape(8, 16, 24)
    for section, axis in TENSOR_SECTIONS.items():
        figure = tensor_slice_figure(array, section, 3)
        np.testing.assert_array_equal(figure.data[0].z, np.take(array, 3, axis=axis).T)
        assert not any(name in figure.layout.title.text for name in ('Axial', 'Sagittal', 'Coronal'))


def test_interactive_history_rejects_incompatible_and_incomplete_manifests(tmp_path):
    run, manifest = research_fixture(tmp_path, 'fixture', (8, 16, 24))
    invalid = run/'interactive/sample-incompatible'; invalid.mkdir()
    write_json(invalid/'manifest.json', dict(manifest, vq_sha256='other', rows=[]))
    incomplete = run/'interactive/sample-incomplete'; incomplete.mkdir()
    (incomplete/'manifest.json').write_text('{')
    folders, rejected = interactive_samples(run, manifest)
    assert {f.name for f in folders} == {'sample-one', 'sample-two'}
    assert set(rejected) == {'sample-incompatible', 'sample-incomplete'}


def test_persistent_history_slice_state_run_switching_and_corruption(tmp_path):
    first, _ = research_fixture(tmp_path, 'first', (8, 16, 24))
    second, _ = research_fixture(tmp_path, 'second', (16, 8, 8))
    app = AppTest.from_function(workspace, args=(str(tmp_path),)).run(timeout=30)
    assert not app.exception
    experiment = next(s for s in app.selectbox if s.label == 'Experiment folder')
    experiment.select_index(experiment.options.index(first.name)).run(timeout=30)
    next(s for s in app.selectbox if s.label == 'Slice orientation').set_value('Row cross-section').run()
    slider = next(s for s in app.slider if s.label == 'Interactive slice')
    assert slider.max == 7
    slider.set_value(7).run()
    next(s for s in app.selectbox if s.label == 'Slice orientation').set_value('Column cross-section').run()
    assert next(s for s in app.slider if s.label == 'Interactive slice').max == 15
    experiment = next(s for s in app.selectbox if s.label == 'Experiment folder')
    experiment.select_index(experiment.options.index(second.name)).run()
    assert not app.exception
    assert next(s for s in app.slider if s.label == 'Interactive slice').max == 7
    restarted = AppTest.from_function(workspace, args=(str(tmp_path),)).run(timeout=30)
    assert not restarted.exception
    assert any(s.label == 'Choose sample to inspect' for s in restarted.selectbox)
    for path in (second/'interactive').glob('*/sample.npy'):
        path.write_bytes(b'corrupted test fixture')
    restarted.run()
    assert not restarted.exception
    assert any('could not be verified' in warning.value for warning in restarted.warning)


def test_display_hash_cache_invalidates_and_reverify_reads_content(labeled_tensors, monkeypatch):
    import app.artifact_cache as cache
    from src.training.common import fingerprint
    cache.clear_verification_cache()
    calls = []
    original = cache.sha256_file
    def counted(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(cache, 'sha256_file', counted)
    expected = fingerprint(labeled_tensors)
    assert cache.dataset_fingerprint(labeled_tensors) == expected
    first_count = len(calls)
    assert first_count == 4
    for _ in range(3):
        assert cache.dataset_fingerprint(labeled_tensors) == expected
    assert len(calls) == first_count
    cache.clear_verification_cache()
    assert cache.dataset_fingerprint(labeled_tensors) == expected
    assert len(calls) == first_count*2
    path = labeled_tensors/'0.npy'
    array = np.load(path); array.flat[0] += .1; np.save(path, array)
    with pytest.raises(ValueError, match='checksum mismatch'):
        cache.dataset_fingerprint(labeled_tensors)
    with pytest.raises(ValueError, match='checksum mismatch'):
        fingerprint(labeled_tensors)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        cache.dataset_fingerprint(labeled_tensors)


def test_display_cache_rechecks_metadata_and_checkpoint_replacement(tmp_path):
    import app.artifact_cache as cache
    cache.clear_verification_cache()
    path = tmp_path/'checkpoint.pt'; path.write_bytes(b'first')
    original = cache.verified_hash(path)
    replacement = tmp_path/'replacement.pt'; replacement.write_bytes(b'other')
    replacement.replace(path)
    assert cache.verified_hash(path) != original
    manifest = tmp_path/'manifest.json'; write_json(manifest, {'value': 1})
    assert cache.read_json(manifest) == {'value': 1}
    write_json(manifest, {'value': 22})
    assert cache.read_json(manifest) == {'value': 22}
