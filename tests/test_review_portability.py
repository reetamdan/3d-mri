import csv
import json
import numpy as np
import pytest
from src.training.review import prepare_review, summarize_review
from src.utils.artifacts import relocate_project, resolve_reference
from src.utils.integrity import sha256_file


def test_review_packet_stays_blinded_and_rejects_unfilled_templates(labeled_tensors, tmp_path):
    generated = tmp_path/'generated'; generated.mkdir()
    rows = []
    for i in range(2):
        path = generated/f'{i}.npy'
        np.save(path, np.random.default_rng(i).uniform(-1, 1, (1, 16, 16, 16)).astype(np.float32))
        rows.append(dict(file=path.name, sha256=sha256_file(path)))
    (generated/'manifest.json').write_text(json.dumps(dict(rows=rows, shape=[16]*3)))
    packet = tmp_path/'review'
    result = prepare_review(labeled_tensors, generated, packet, count=2)
    assert result['cases'] == 4 and not result['scores_collected']
    with (packet/'review-template.csv').open(newline='') as f:
        ratings = list(csv.DictReader(f))
    assert all(not r['anatomy'] for r in ratings)
    assert {p.stem for p in (packet/'cases').glob('*.npy')} == {r['case_id'] for r in ratings}
    with pytest.raises(ValueError, match='reviewer IDs'):
        summarize_review(packet, packet/'review-template.csv', tmp_path/'fake-results.json')
    # Explicitly artificial form-validation ratings, never medical study results.
    artificial = [dict(r, reviewer_id=name, appearance='3', anatomy='2', continuity='2', suspected_synthetic='1')
                  for name in ('fixture-reviewer-a', 'fixture-reviewer-b') for r in ratings]
    path = tmp_path/'artificial-test-ratings.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(artificial[0])); writer.writeheader(); writer.writerows(artificial)
    report = summarize_review(packet, path, tmp_path/'test-only-summary.json')
    assert len(report['reviewers']) == 2 and not report['qualifications_verified_by_software']


def test_legacy_path_relocation_keeps_checkpoint_bytes_and_external_paths(tmp_path):
    project = tmp_path/'project'
    (project/'configs').mkdir(parents=True); (project/'src').mkdir()
    data = project/'data/processed/cohort'; data.mkdir(parents=True)
    run = project/'runs/example'; run.mkdir(parents=True)
    checkpoint = run/'weights.pt'; checkpoint.write_bytes(b'explicit fixture; not a model')
    digest = sha256_file(checkpoint)
    old = 'Z:/unavailable/developer/project'
    report = run/'report.json'
    report.write_text(json.dumps(dict(input_dataset=old+'/data/processed/cohort',
                                     vq_checkpoint=old+'/runs/example/weights.pt',
                                     external='/unknown/outside/project.npy')))
    assert resolve_reference(run, old+'/data/processed/cohort') == data
    result = relocate_project(project)
    assert result['count'] == 1
    values = json.loads(report.read_text())
    assert values['input_dataset'] == '../../data/processed/cohort'
    assert resolve_reference(run, values['vq_checkpoint']) == checkpoint
    assert values['external'] == '/unknown/outside/project.npy'
    assert sha256_file(checkpoint) == digest
    assert relocate_project(project)['count'] == 0
