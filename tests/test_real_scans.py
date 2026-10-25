"""Offline integration checks on explicitly downloaded public MRI samples.

Skipped on clean clones. No test downloads any data.
"""
from pathlib import Path
import json
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from src.utils.config import load_config
from src.data.loader import load_mri
from src.data.pipeline import inspect_dataset, run_pipeline
from src.data.preprocessing import preprocess_volume
from src.data.dataset import MRIDataset
from src.metrics import downsample_reconstruction_baseline
from src.visualization.mri_viewer import canonical_volume, slice_figure, PLANES
from src.visualization.volume_viewer import volume_figure

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'data/metadata/sample_sources.json'
pytestmark = pytest.mark.skipif(not SOURCES.exists(), reason='Run scripts/download_sample_data.py to enable real MRI tests')


def sample_records():
    return json.loads(SOURCES.read_text())['samples']


@pytest.mark.parametrize('sample_index', [0, 1, 2])
def test_real_volume_views_and_preprocessing(sample_index):
    sample = sample_records()[sample_index]
    volume, meta = load_mri(ROOT / sample['nifti_path'])
    config = load_config(ROOT / 'configs/real-scans.yaml')
    output, transformed = preprocess_volume(volume, meta, config['preprocessing'])
    assert output.shape == (128, 128, 128)
    assert output.min() == pytest.approx(-1)
    assert output.max() == pytest.approx(1)
    assert np.isfinite(output).all()
    canonical, affine = canonical_volume(volume, meta['affine'])
    for plane, axis in PLANES.items():
        for index in (0, canonical.shape[axis]//2, canonical.shape[axis]-1):
            fig = slice_figure(canonical, plane, index)
            assert np.asarray(fig.data[0].z).size > 0
    assert len(volume_figure(output, transformed['affine']).data[0].value) <= 48**3
    _, baseline = downsample_reconstruction_baseline(output)
    assert np.isfinite(list(baseline.values())).all()
    assert 0 < baseline['ssim'] <= 1


def test_real_export_reproducibility_and_tensor_roundtrip(tmp_path):
    config = load_config(ROOT / 'configs/real-scans.yaml')
    config['dataset']['processed_dir'] = str(tmp_path / 'processed')
    config['dataset']['metadata_dir'] = str(tmp_path / 'metadata')
    config['logs_dir'] = str(tmp_path / 'logs')
    valid, errors, report = inspect_dataset(config)
    assert len(valid) == 3 and errors == [] and report['subjects'] == 2
    first = run_pipeline(config)
    config['export_format'] = 'pt'
    second = run_pipeline(config)
    assert len(first['rows']) == len(second['rows']) == 3
    assert first['errors'] == second['errors'] == []
    seen = {}
    for row in first['rows']:
        seen.setdefault(row['patient_id'], set()).add(row['split'])
    assert len(seen) == 2 and all(len(s) == 1 for s in seen.values())
    for split in ('train', 'validation', 'test'):
        npy = MRIDataset(first['output_dir'], split)
        pt = MRIDataset(second['output_dir'], split)
        assert len(npy) == len(pt)
        for i in range(len(npy)):
            np.testing.assert_array_equal(npy[i].numpy(), pt[i].numpy())


def test_real_dashboard_all_scans_and_controls():
    app = AppTest.from_file(str(ROOT / 'app/streamlit_app.py')).run(timeout=60)
    app.switch_page('pages/workbench.py').run(timeout=60)
    next(t for t in app.text_input if t.label == 'Configuration file').set_value(str(ROOT / 'configs/real-scans.yaml')).run(timeout=60)
    next(b for b in app.button if b.label == 'Index & validate dataset').click().run(timeout=60)
    assert not app.exception
    assert app.metric[0].value == '3'
    app.selectbox(key='data_tool').select('MRI Viewer').run(timeout=60)
    scan_widget = next(s for s in app.selectbox if s.label == 'Scan')
    for scan in scan_widget.options:
        next(s for s in app.selectbox if s.label == 'Scan').select_index(scan_widget.options.index(scan)).run(timeout=60)
        assert not app.exception
        for slider in app.slider:
            slider.set_value(slider.max)
        app.run(timeout=60)
        assert not app.exception
    app.checkbox[0].check().run(timeout=60)
    assert not app.exception
    app.selectbox(key='data_tool').select('Preprocessing').run(timeout=60)
    next(b for b in app.button if b.label == 'Run preprocessing preview').click().run(timeout=60)
    assert not app.exception
    assert app.session_state['preview'][1].shape == (128, 128, 128)
    for plane in PLANES:
        next(s for s in app.selectbox if s.label == 'Comparison plane').set_value(plane).run(timeout=60)
        assert not app.exception
    app.selectbox(key='data_tool').select('Intensity Analysis').run(timeout=60)
    next(b for b in app.button if b.label == 'Run preprocessing preview').click().run(timeout=60)
    next(b for b in app.button if b.label == 'Calculate interpolation reconstruction baseline').click().run(timeout=60)
    assert not app.exception
    app.selectbox(key='data_tool').select('Dataset Quality').run(timeout=60)
    assert not app.exception
    assert app.metric[2].value == '0'
