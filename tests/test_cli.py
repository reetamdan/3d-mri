import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_installable_cli_works_outside_repository(tmp_path):
    command = [sys.executable, '-B', '-m', 'mri3d', '--help']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and 'freeze-protocol' in result.stdout and 'relocate' in result.stdout


def test_feature_cli_records_input_hashes_and_does_not_download_weights(tmp_path):
    a, b = tmp_path/'real.npy', tmp_path/'generated.npy'
    np.save(a, np.array([[0.], [1.]])); np.save(b, np.array([[2.], [3.]]))
    output = tmp_path/'metrics.json'
    result = subprocess.run([sys.executable, '-B', '-m', 'mri3d', 'evaluate-features',
        '--real', str(a), '--generated', str(b), '--extractor-id', 'closed-form-fixture-not-medical',
        '--output', str(output)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['frechet_feature_distance'] == 4.
    assert len(report['real_features_sha256']) == 64
    assert 'exam-level' in report['protocol']
