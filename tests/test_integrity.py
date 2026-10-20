import csv
import json
import numpy as np
import pytest
from src.data.dataset import MRIDataset
from src.training.common import fingerprint
from src.training.evaluation import segmentation_overlap
from src.training.downstream import classification_metrics
from src.utils.integrity import generated_volume, sha256_file, confined_path
from app.research_workspace import research_runs
from pathlib import Path
from src.training.common import model_config
from src.training.experiments import train_vq


def test_tensor_contents_and_metadata_integrity(tmp_path):
    tensor = tmp_path/'scan.npy'
    np.save(tensor, np.zeros((1, 8, 8, 8), dtype=np.float32))
    row = dict(scan_id='scan', patient_id='patient', split='train', output_file='scan.npy')
    def metadata(rows):
        with (tmp_path/'metadata.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    metadata([row])
    config = model_config(Path(__file__).resolve().parents[1]/'configs/smoke-models.yaml')
    config['smoke_test'] = False
    with pytest.raises(ValueError, match='validation split'):
        train_vq(tmp_path, tmp_path/'rejected-run', config)
    assert not (tmp_path/'rejected-run').exists()
    before = fingerprint(tmp_path)
    np.save(tensor, np.ones((1, 8, 8, 8), dtype=np.float32))
    assert fingerprint(tmp_path) != before
    row['output_sha256'] = sha256_file(tensor)
    metadata([row])
    assert MRIDataset(tmp_path)[0].sum() == 512
    np.save(tensor, np.zeros((1, 8, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match='checksum'):
        fingerprint(tmp_path)
    with pytest.raises(ValueError, match='checksum'):
        MRIDataset(tmp_path)[0]
    metadata([row, dict(row, scan_id='another', split='validation')])
    with pytest.raises(ValueError, match='leakage'):
        MRIDataset(tmp_path)


def test_generated_integrity_and_paths(tmp_path):
    path = tmp_path/'sample.npy'
    np.save(path, np.zeros((1, 8, 8, 8), dtype=np.float32))
    row = dict(file=path.name, sha256=sha256_file(path))
    assert generated_volume(tmp_path, row, [8, 8, 8]).shape == (1, 8, 8, 8)
    np.save(path, np.ones((1, 8, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match='checksum'):
        generated_volume(tmp_path, row, [8, 8, 8])
    with pytest.raises(ValueError, match='escapes'):
        confined_path(tmp_path, '../outside.npy')


def test_metrics_reject_invalid_ground_truth():
    with pytest.raises(ValueError, match='binary'):
        segmentation_overlap([float('nan')], [1])
    with pytest.raises(ValueError, match='binary'):
        classification_metrics([[0.5, 1, 0]], [[0.5, 0.5, 0.5]])


def test_dashboard_excludes_smoke_and_corrupted_checkpoints(tmp_path):
    folder = tmp_path/'run'/'vq'
    folder.mkdir(parents=True)
    checkpoint = folder/'last.pt'
    checkpoint.write_bytes(b'test-only artifact')
    info = dict(smoke_test=True, completed_epochs=1, checkpoint_sha256=sha256_file(checkpoint))
    manifest = folder/'run.json'
    manifest.write_text(json.dumps(info))
    assert research_runs(tmp_path) == []
    info['smoke_test'] = False
    manifest.write_text(json.dumps(info))
    assert research_runs(tmp_path) == [folder.parent]
    checkpoint.write_bytes(b'altered')
    assert research_runs(tmp_path) == []
