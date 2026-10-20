from pathlib import Path
import shutil
import yaml
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app/streamlit_app.py')


def test_empty_app_launch():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == 'Dataset Overview'


def test_research_workspace_pages():
    app = AppTest.from_file(APP).run(timeout=30)
    for page in ('MRNet Dataset', 'Model Experiments'):
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception


def test_dashboard_workflow(config, nifti, tmp_path):
    folder = Path(config['dataset']['raw_dir']) / 'synthetic-test-subject'
    folder.mkdir()
    shutil.copyfile(nifti, folder / 'test-only.nii.gz')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    app = AppTest.from_file(APP).run(timeout=30)
    app.sidebar.text_input[0].set_value(str(path)).run()
    app.sidebar.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == '1'
    for page in ['MRI Viewer', 'Preprocessing', 'Intensity Analysis', 'Dataset Quality', 'Export']:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, page
        if page == 'MRI Viewer':
            app.checkbox[0].check().run(timeout=30)
            assert not app.exception
        if page in ('Preprocessing', 'Intensity Analysis'):
            next(b for b in app.button if b.label == 'Run preprocessing preview').click().run(timeout=30)
            assert not app.exception
            next(b for b in app.button if b.label == 'Calculate interpolation reconstruction baseline').click().run(timeout=30)
            assert not app.exception
        if page == 'Export':
            next(b for b in app.button if b.label == 'Process & export dataset').click().run(timeout=30)
            assert not app.exception
            assert app.session_state['run']['report']['valid_scans'] == 1


@pytest.mark.parametrize('settings', [
    {'target_shape': [2048, 8, 8]},
    {'crop_size': [2048, 8, 8]},
    {'target_spacing': [.05, 1., 1.]},
    {'target_spacing': [25., 1., 1.]},
])
def test_valid_config_values_outside_default_preview_bounds(config, nifti, tmp_path, settings):
    folder = Path(config['dataset']['raw_dir'])/'test-only-subject'
    folder.mkdir()
    shutil.copyfile(nifti, folder/'test-only.nii.gz')
    config['preprocessing'].update(settings)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    app = AppTest.from_file(APP).run(timeout=30)
    app.sidebar.text_input[0].set_value(str(path)).run(timeout=30)
    app.sidebar.button[0].click().run(timeout=30)
    app.sidebar.radio[0].set_value('Preprocessing').run(timeout=30)
    assert not app.exception
    controls = {n.label: n.value for n in app.number_input}
    if 'target_shape' in settings:
        assert controls['Target R'] == 2048
    elif 'crop_size' in settings:
        assert controls['Crop R'] == 2048
    else:
        assert controls['Spacing R (mm)'] == settings['target_spacing'][0]


def test_failed_reindex_clears_previous_dataset_preview_and_export(config, nifti, tmp_path):
    folder = Path(config['dataset']['raw_dir'])/'test-only-subject'
    folder.mkdir()
    shutil.copyfile(nifti, folder/'test-only.nii.gz')
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    app = AppTest.from_file(APP).run(timeout=30)
    app.sidebar.text_input[0].set_value(str(path)).run(timeout=30)
    app.sidebar.button[0].click().run(timeout=30)
    app.sidebar.radio[0].set_value('Preprocessing').run(timeout=30)
    next(b for b in app.button if b.label == 'Run preprocessing preview').click().run(timeout=30)
    app.sidebar.radio[0].set_value('Export').run(timeout=30)
    next(b for b in app.button if b.label == 'Process & export dataset').click().run(timeout=30)
    assert not app.exception and 'preview' in app.session_state and 'run' in app.session_state
    shutil.rmtree(config['dataset']['raw_dir'])
    app.sidebar.button[0].click().run(timeout=30)
    assert not app.exception and app.error
    for key in ('records', 'errors', 'report', 'preview', 'run'):
        assert key not in app.session_state
    assert not any(b.label == 'Process & export dataset' for b in app.button)
