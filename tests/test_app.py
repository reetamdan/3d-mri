from pathlib import Path
import shutil
import yaml
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
