from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import pytest
from src.utils.config import load_config


def pytest_configure(config):
    # pytest creates its chosen basetemp, but not a missing parent directory.
    if config.option.basetemp:
        Path(config.option.basetemp).resolve().parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def synthetic_volume():
    """Synthetic numerical test fixture, not an MRI or clinical image."""
    x, y, z = np.indices((12, 14, 10), dtype=np.float32)
    return x + 2*y + 3*z


@pytest.fixture
def nifti(tmp_path, synthetic_volume):
    path = tmp_path / 'test_only.nii.gz'
    img = nib.Nifti1Image(synthetic_volume, np.diag([2., 1., 3., 1.]))
    img.header.set_xyzt_units('mm')
    nib.save(img, path)
    return path


@pytest.fixture
def config(tmp_path):
    c = load_config(Path(__file__).resolve().parents[1] / 'configs/prd-example.yaml')
    for key, name in [('raw_dir', 'raw'), ('processed_dir', 'processed'), ('metadata_dir', 'metadata')]:
        c['dataset'][key] = str(tmp_path / name)
    Path(c['dataset']['raw_dir']).mkdir()
    c['logs_dir'] = str(tmp_path / 'logs')
    c['preprocessing'].update(target_shape=[16, 16, 8], crop_size=[18, 18, 10])
    return c


@pytest.fixture
def labeled_tensors(tmp_path):
    """Artificial tensors/labels for software tests, never a research cohort."""
    from src.utils.logging import write_csv
    from src.utils.integrity import sha256_file
    root = tmp_path/'labeled-tensors'
    root.mkdir()
    rows = []
    for i, split in enumerate(('train', 'train', 'validation', 'validation')):
        path = root/f'{i}.npy'
        np.save(path, np.random.default_rng(i).uniform(-1, 1, (1, 16, 16, 16)).astype(np.float32))
        rows.append(dict(scan_id=f'fixture-{i}', patient_id=f'fixture-patient-{i}', split=split,
                         labels=[i % 2]*3, output_file=path.name, output_sha256=sha256_file(path)))
    write_csv(root/'metadata.csv', rows, list(rows[0]))
    return root
