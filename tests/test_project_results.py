"""Artificial UI artifacts exercise comparison integrity; never a research cohort."""
from copy import deepcopy
import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.project_results import published_results, verified_run, image_pair
from src.training.common import fingerprint, TrainingDataset
from src.utils.integrity import sha256_file
from src.utils.logging import write_json


def page(root):
    from pathlib import Path
    from app.project_results import project_results_page
    project_results_page(Path(root))


@pytest.fixture
def result_workspace(labeled_tensors, tmp_path):
    root = tmp_path/'workspace'; root.mkdir()
    study = root/'runs/ARTIFICIAL-UI-FIXTURE'
    cohort = fingerprint(labeled_tensors)
    data = TrainingDataset(labeled_tensors, 'validation', [16]*3)
    records, groups = [], {}
    for arm, value in (('ae_vq_baseline', .1), ('ae_kl_aniso', .05)):
        name = f'{arm}-42'; folder = study/name/'reconstruction'; folder.mkdir(parents=True)
        checkpoint = study/name/'training/best.pt'; checkpoint.parent.mkdir()
        checkpoint.write_bytes(f'Not a model: isolated UI-only fixture {arm}'.encode())
        metrics = dict(psnr=25.+value, ssim=.8, ms_ssim=.9, gradient_energy_ratio=.4,
                       high_frequency_energy_ratio=.2, edge_mae=.1)
        rows = []
        for i, source in enumerate(data.rows):
            path = folder/f'{i}.npy'
            np.save(path, (data[i][0].numpy()+value).clip(-1, 1))
            rows.append(dict(scan_id=source['scan_id'], file=path.name, sha256=sha256_file(path), **metrics))
        record = dict(name=name, arm=arm, seed=42, checkpoint=f'{name}/training/best.pt',
                      checkpoint_sha256=sha256_file(checkpoint), quantitative_gate={'detail': False})
        records.append(record)
        write_json(folder/'report.json', dict(rows=rows, means=metrics, shape=[16]*3, latent_shape=[2, 4, 4, 4],
            scalar_compression=32, split='validation', input_dataset=str(labeled_tensors),
            dataset_fingerprint=cohort, checkpoint_sha256=record['checkpoint_sha256'], smoke_test=False))
        write_json(checkpoint.parent/'history.json', [dict(epoch=1, l1=.1, validation_l1=.2)])
        groups[arm] = dict(seeds=[42], spatial_compression=[4]*3,
            metrics={key: dict(mean=number, seed_sd=None) for key, number in metrics.items()},
            train_peak_mib=1., gates_passed=0)
    write_json(study/'study.json', dict(status='completed', smoke_test=False,
        plan=dict(dataset_fingerprint=cohort), records=records))
    summary = dict(groups=groups, records=[dict(r, source_study='runs/ARTIFICIAL-UI-FIXTURE') for r in records],
                   dataset_fingerprint=cohort)
    write_json(root/'docs/results/ae_bakeoff_measured.json', summary)
    return root, summary


def test_project_comparison_uses_identical_cases_and_detects_stale_models_and_images(result_workspace):
    root, summary = result_workspace
    record = summary['records'][1]
    # Published Windows-style references also work when checked on Linux CI.
    record['source_study'] = record['source_study'].replace('/', '\\')
    run = verified_run(root, record, summary['dataset_fingerprint'])
    identity = run.report['rows'][0]['scan_id']
    original, restored, _ = image_pair(run, identity)
    assert original.shape == restored.shape == (16, 16, 16)
    np.testing.assert_allclose(restored, (original+.05).clip(-1, 1))
    (run.folder/run.report['rows'][0]['file']).write_bytes(b'corrupted UI fixture')
    with pytest.raises(ValueError, match='checksum'):
        image_pair(run, identity)
    run.checkpoint.write_bytes(b'changed model')
    with pytest.raises(ValueError, match='checkpoint checksum'):
        verified_run(root, record, summary['dataset_fingerprint'])


def test_project_summary_and_study_reject_artificial_or_unfinished_results(result_workspace):
    root, summary = result_workspace
    altered = deepcopy(summary); altered['records'][0]['smoke_test'] = True
    write_json(root/'docs/results/ae_bakeoff_measured.json', altered)
    with pytest.raises(ValueError, match='Artificial'):
        published_results(root)
    path = root/'runs/ARTIFICIAL-UI-FIXTURE/study.json'
    study = json.loads(path.read_text()); study['status'] = 'in_progress'; write_json(path, study)
    with pytest.raises(ValueError, match='completed'):
        verified_run(root, summary['records'][0], summary['dataset_fingerprint'])
    study['status'] = 'completed'; study.pop('smoke_test')
    study['plan']['configs'] = {'original_format': {'smoke_test': False}}
    write_json(path, study)
    assert verified_run(root, summary['records'][0], summary['dataset_fingerprint']).report['rows']
    study['plan']['configs']['original_format']['smoke_test'] = True; write_json(path, study)
    with pytest.raises(ValueError, match='completed'):
        verified_run(root, summary['records'][0], summary['dataset_fingerprint'])


def test_project_results_controls_views_errors_and_missing_artifacts(result_workspace):
    root, _ = result_workspace
    app = AppTest.from_function(page, args=(str(root),)).run(timeout=30)
    assert not app.exception and not app.error
    assert any(s.label == 'Validation exam' for s in app.selectbox)
    next(s for s in app.selectbox if s.label == 'View direction').set_value('Row cross-section').run()
    assert next(s for s in app.slider if s.label == 'Synchronized MRI slice').max == 15
    next(c for c in app.checkbox if c.label == 'Show absolute error maps').check().run()
    next(c for c in app.checkbox if c.label == 'Include interpolation reference').check().run()
    assert not app.exception
    next(s for s in app.selectbox if s.label == 'Reconstruction model').set_value('ae_vq_baseline').run()
    assert not app.exception
    empty = AppTest.from_function(page, args=(str(root/'empty'),)).run(timeout=30)
    assert not empty.exception and any('No completed benchmark' in i.value for i in empty.info)
