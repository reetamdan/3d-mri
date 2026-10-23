"""Descriptor/report/CLI contracts on explicitly artificial, isolated artifacts."""
import json
import sys
import numpy as np
import pytest
import torch

from src.training.common import TrainingDataset, fingerprint
from src.utils.integrity import sha256_file
from src.utils.logging import write_json


def test_feature_audit_detects_exact_copies_and_binds_paired_model_data(labeled_tensors, tmp_path, monkeypatch):
    from scripts import evaluate_representation_features as command
    train = TrainingDataset(labeled_tensors, 'train', [16]*3)
    valid = TrainingDataset(labeled_tensors, 'validation', [16]*3)
    descriptor_dir = tmp_path/'descriptor'; descriptor_dir.mkdir()
    descriptor = descriptor_dir/'features.pt'
    toy = torch.nn.Sequential(torch.nn.AdaptiveAvgPool3d((2, 2, 2)), torch.nn.Flatten())
    torch.jit.trace(toy, torch.zeros(1, 1, 16, 16, 16)).save(str(descriptor))
    write_json(descriptor_dir/'extractor.json', dict(feature_sha256=sha256_file(descriptor),
               architecture='Artificial untrained numerical fixture, not a medical model'))
    generated = tmp_path/'generated'; generated.mkdir()
    generated_rows = []
    for i in range(len(train)):
        path = generated/f'{i}.npy'; np.save(path, train[i][0].numpy())
        generated_rows.append(dict(file=path.name, sha256=sha256_file(path), conditioning_labels=[i]*3))
    data_hash = fingerprint(labeled_tensors)
    write_json(generated/'manifest.json', dict(rows=generated_rows, shape=[16]*3,
               dataset_fingerprint=data_hash, smoke_test=True))
    study = tmp_path/'study'; folder = study/'fixture/reconstruction'; folder.mkdir(parents=True)
    checkpoint = study/'fixture/weights.pt'; checkpoint.write_bytes(b'Explicit non-model lineage fixture')
    paired_rows = []
    for i, row in enumerate(valid.rows):
        path = folder/f'{i}.npy'; np.save(path, valid[i][0].numpy())
        paired_rows.append(dict(scan_id=row['scan_id'], file=path.name, sha256=sha256_file(path)))
    write_json(folder/'report.json', dict(rows=paired_rows, checkpoint_sha256=sha256_file(checkpoint), dataset_fingerprint=data_hash))
    write_json(study/'study.json', dict(status='completed', plan=dict(dataset_fingerprint=data_hash,
        configs={'fixture': dict(shape=[16]*3, smoke_test=True)}),
        records=[dict(name='fixture', arm='fixture', seed=42, checkpoint='fixture/weights.pt', checkpoint_sha256=sha256_file(checkpoint))]))
    output = tmp_path/'audit'
    argv = ['feature-audit', '--study', str(study), '--data', str(labeled_tensors), '--generated', str(generated),
            '--extractor', str(descriptor), '--output', str(output), '--device', 'cpu']
    monkeypatch.setattr(sys, 'argv', argv); command.main()
    report = json.loads((output/'feature-memorization-report.json').read_text())
    assert report['smoke_test']
    assert all(r['exact_file_duplicate'] and r['nearest_ssim'] == 1 for r in report['generated_neighbors'])
    assert all(r['mean_paired_feature_cosine_distance'] < 1e-6 for r in report['paired_reconstruction'])
    checkpoint.write_bytes(b'changed')
    monkeypatch.setattr(sys, 'argv', argv[:-3]+[str(tmp_path/'bad'), '--device', 'cpu'])
    with pytest.raises(ValueError, match='stale model/data lineage'):
        command.main()


def test_measured_tables_reject_artificial_and_incomplete_runs(tmp_path):
    from scripts.summarize_representation import summarize
    root = tmp_path/'study'; root.mkdir()
    write_json(root/'study.json', dict(status='completed', plan=dict(dataset_fingerprint='fixture',
        configs={'fixture': dict(smoke_test=True)}), records=[]))
    with pytest.raises(ValueError, match='Artificial smoke'):
        summarize([root], tmp_path/'not-results.md')
    write_json(root/'study.json', dict(status='completed', plan=dict(dataset_fingerprint='fixture',
        configs={'fixture': dict(smoke_test=False, epochs=2, steps_per_epoch=1, batch_size=2, gradient_accumulation=1, shape=[16]*3)}),
        records=[dict(arm='fixture', name='fixture-42', seed=42)]))
    write_json(root/'fixture-42/training/history.json', [dict(optimizer_steps=1)])
    with pytest.raises(ValueError, match='Incomplete optimization budget'):
        summarize([root], tmp_path/'not-results.md')


def test_local_pretraining_cli_filters_acquisition_and_does_not_assign_pathology(tmp_path, monkeypatch, capsys):
    h5py = pytest.importorskip('h5py')
    from scripts import prepare_knee_pretraining as command
    from src.utils.logging import write_csv
    source = tmp_path/'source'; source.mkdir()
    with h5py.File(source/'image.h5', 'w') as file:
        file.create_dataset('reconstruction_esc', data=np.arange(5*8*8).reshape(5, 8, 8))
        file.attrs['acquisition'] = 'CORPD_FBK'
    manifest = tmp_path/'manifest.csv'
    write_csv(manifest, [dict(file_path='image.h5', split='train', patient_id='', acquisition='CORPD_FBK')],
              ['file_path', 'split', 'patient_id', 'acquisition'])
    output = tmp_path/'processed'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--source', str(source), '--manifest', str(manifest), '--output', str(output),
        '--dataset', 'fastmri', '--acquisition', 'CORPD_FBK', '--image-key', 'reconstruction_esc', '--shape', '8', '8', '8'])
    command.main()
    assert json.loads(capsys.readouterr().out)['valid_scans'] == 1
    rows = TrainingDataset(output, 'train', [8]*3).rows
    assert rows[0]['labels'] == '' and rows[0]['patient_id'] == ''
