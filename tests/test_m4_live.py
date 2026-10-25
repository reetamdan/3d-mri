"""Local real-checkpoint acceptance test. Never downloads or trains on import."""
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from src.utils.integrity import generated_volume, sha256_file
from src.utils.artifacts import resolve_reference

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/M4-MRNET-20261021'
pytestmark = pytest.mark.skipif(not (RUN/'evidence.json').exists(), reason='Complete scripts/run_m4.py for real-checkpoint acceptance')


def test_real_trained_dashboard_inference_and_new_seed(tmp_path):
    run = tmp_path/'runs'/RUN.name
    shutil.copytree(RUN, run, ignore=shutil.ignore_patterns('interactive', 'sampling-ablation', 'seed-repeat', 'evidence-assets'))
    report_path = run/'reconstruction/report.json'
    report = json.loads(report_path.read_text())
    report['input_dataset'] = str(resolve_reference(RUN/'reconstruction', report['input_dataset']))
    report_path.write_text(json.dumps(report))
    app = AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=60)
    app.switch_page('pages/workbench.py').run(timeout=60)
    app.selectbox(key='data_tool').select('Legacy model audit').run(timeout=60)
    next(t for t in app.text_input if t.label == 'Research runs directory').set_value(str(run.parent)).run(timeout=60)
    assert not app.exception
    experiment = next(s for s in app.selectbox if s.label == 'Experiment folder')
    experiment.select_index(next(i for i, option in enumerate(experiment.options) if option == RUN.name)).run(timeout=60)
    assert not app.exception
    assert app.metric[0].value == '96 / 24'
    next(b for b in app.button if b.label == 'Reconstruct selected input now').click().run(timeout=60)
    assert not app.exception
    assert any('reconstruction inference completed' in str(s.value).lower() for s in app.success)
    next(n for n in app.number_input if n.label == 'Sampling seed').set_value(2037).run(timeout=60)
    next(b for b in app.button if b.label == 'Generate a new volume').click().run(timeout=60)
    assert not app.exception
    live = app.session_state['live_generation']
    assert Path(live['run']) == run
    folder = Path(live['folder'])
    assert folder.is_relative_to(run/'interactive')
    manifest = json.loads((folder/'manifest.json').read_text())
    row = manifest['rows'][0]
    assert row['seed'] == 2037 and not manifest['smoke_test']
    array = generated_volume(folder, row, manifest['shape'])
    assert array.shape == (1, 128, 128, 32) and np.std(array) > 0
    expected = json.loads((RUN/'vq/run.json').read_text())
    assert manifest['vq_sha256'] == expected['checkpoint_sha256']
    saved = json.loads((RUN/'generated/manifest.json').read_text())['rows'][0]
    assert sha256_file(folder/row['latent_file']) != saved['latent_sha256']
