"""Exercise a built wheel from an unrelated directory, without editable imports."""
from pathlib import Path
import subprocess
import sys
import zipfile

from mri3d.presets import PRESETS
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_build_hook_requires_no_application_imports(monkeypatch):
    import builtins
    import importlib.util
    original = builtins.__import__
    def isolated_import(name, *args, **kwargs):
        if name.split('.')[0] in ('mri3d', 'src', 'app', 'scripts'):
            raise ModuleNotFoundError('Application unavailable in isolated build')
        return original(name, *args, **kwargs)
    spec = importlib.util.spec_from_file_location('isolated_build_hook', ROOT/'mri3d/_build.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(builtins, '__import__', isolated_import)
    spec.loader.exec_module(module)
    assert module.PRESETS == PRESETS


def test_default_preprocessing_paths_belong_to_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert Path(config['dataset']['raw_dir']) == tmp_path/'data/raw'
    assert Path(config['logs_dir']) == tmp_path/'logs'


def test_built_wheel_has_presets_and_runs_outside_checkout(tmp_path):
    wheels, installed, outside = (tmp_path/name for name in ('wheels', 'installed', 'outside'))
    outside.mkdir()
    build = subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(ROOT), '--no-deps', '--no-build-isolation',
                            '--no-index', '--wheel-dir', str(wheels)], cwd=outside, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout+build.stderr
    wheel, = wheels.glob('*.whl')
    with zipfile.ZipFile(wheel) as archive:
        resources = {n.rsplit('/', 1)[-1] for n in archive.namelist() if n.startswith('mri3d/presets/') and n.endswith('.yaml')}
    assert resources == set(PRESETS)
    install = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--target', str(installed), str(wheel)],
                             cwd=outside, capture_output=True, text=True, timeout=60)
    assert install.returncode == 0, install.stdout+install.stderr
    script = '''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import mri3d, src.training.common, scripts.research
from mri3d.configuration import read_configuration, workspace_root
from mri3d.presets import PRESETS
from src.utils.config import load_config
for module in (mri3d, src.training.common, scripts.research):
    assert Path(module.__file__).is_relative_to(Path(sys.argv[1])), module.__file__
for name in PRESETS:
    assert isinstance(read_configuration(preset=name), dict)
assert src.training.common.model_config()['threads'] == 4
assert workspace_root() == Path.cwd()
assert Path(load_config()['dataset']['raw_dir']) == Path.cwd()/'data/raw'
from src.utils.logging import write_csv
from src.utils.integrity import sha256_file
from src.training.experiments import train_vq, evaluate_reconstruction
import numpy as np
data = Path('fixture'); data.mkdir()
rows = []
for i, split in enumerate(('train', 'validation')):
    path = data/f'{i}.npy'
    np.save(path, np.random.default_rng(i).uniform(-1, 1, (1, 8, 8, 8)).astype(np.float32))
    rows.append(dict(scan_id=str(i), patient_id=f'p{i}', split=split, output_file=path.name, output_sha256=sha256_file(path)))
write_csv(data/'metadata.csv', rows, list(rows[0]))
c = src.training.common.model_config(preset='smoke-models.yaml')
c.update(shape=[8]*3, epochs=1, steps_per_epoch=1, gan_weight=0)
checkpoint = train_vq(data, 'training', c)
report = evaluate_reconstruction(data, checkpoint, 'evaluation', device='cpu')
assert len(report['rows']) == 1
sys.argv = ['mri3d', 'show-config', '--preset', 'models.yaml']
scripts.research.main()
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, str(installed)], cwd=outside,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '"threads": 4' in result.stdout
