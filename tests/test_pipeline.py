import json
from pathlib import Path
import shutil
import pytest
from src.data.pipeline import run_pipeline, inspect_dataset
from src.data.indexing import index_dataset
from src.data.splitting import split_patients
from src.data.dataset import MRIDataset
from src.metrics import reconstruction_metrics
from src.utils.config import validate_config


def test_subject_splits():
    patients = [f's{i}' for i in range(20)] * 2
    ratios = dict(train=.7, validation=.15, test=.15)
    a = split_patients(patients, ratios, 42)
    assert a == split_patients(list(reversed(patients)), ratios, 42)
    assert [list(a.values()).count(s) for s in ratios] == [14, 3, 3]
    groups = [{p for p, s in a.items() if s == split} for split in ratios]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


@pytest.mark.parametrize('fmt', ['npy', 'pt'])
def test_export_and_bad_file_isolation(config, nifti, fmt):
    raw = Path(config['dataset']['raw_dir'])
    for subject in range(5):
        folder = raw / f'subject{subject}'
        folder.mkdir()
        for scan in range(2):
            shutil.copyfile(nifti, folder / f'scan{scan}.nii.gz')
    (raw / 'subject0' / 'bad.nii.gz').write_bytes(b'broken')
    config['export_format'] = fmt
    result = run_pipeline(config)
    assert result['report']['valid_scans'] == 10
    assert result['report']['invalid_scans'] == 1
    run = Path(result['output_dir'])
    assignments = {}
    for row in result['rows']:
        assignments.setdefault(row['patient_id'], set()).add(row['split'])
        assert (run / row['output_file']).exists()
    assert all(len(s) == 1 for s in assignments.values())
    dataset = MRIDataset(run)
    assert tuple(dataset[0].shape) == (1, 16, 16, 8)
    assert json.loads((run / 'experiment.json').read_text())['config']['seed'] == 42
    assert (Path(config['logs_dir']) / 'validation_errors.csv').exists()
    second = run_pipeline(config)
    assert second['output_dir'] != result['output_dir']
    assert [r['split'] for r in second['rows']] == [r['split'] for r in result['rows']]


def test_manifest_missing_and_grouping(config, nifti):
    raw = Path(config['dataset']['raw_dir'])
    shutil.copyfile(nifti, raw / 'scan.nii.gz')
    with pytest.raises(ValueError, match='manifest'):
        index_dataset(config['dataset'])
    manifest = raw.parent / 'manifest.csv'
    manifest.write_text('file_path,patient_id\nscan.nii.gz,subject-a\nmissing.nii.gz,subject-a\n')
    config['dataset']['manifest'] = str(manifest)
    indexed = index_dataset(config['dataset'])
    assert indexed[0]['patient_id'] == indexed[1]['patient_id']
    valid, errors, report = inspect_dataset(config)
    assert len(valid) == len(errors) == 1
    assert errors[0]['code'] == 'missing'


def test_metrics(synthetic_volume):
    metrics = reconstruction_metrics(synthetic_volume, synthetic_volume, data_range=64)
    assert metrics['mse'] == metrics['mae'] == 0
    assert metrics['ssim'] == pytest.approx(1)
    assert metrics['psnr'] == float('inf')
    changed = reconstruction_metrics(synthetic_volume, synthetic_volume+1, data_range=64)
    assert changed['mse'] == pytest.approx(1)


def test_bad_config(config):
    config['split']['train'] = .9
    with pytest.raises(ValueError, match='ratios'):
        validate_config(config)
