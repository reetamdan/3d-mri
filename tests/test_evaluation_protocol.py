import csv
import json
import numpy as np
import pytest
from src.training.common import fingerprint
from src.training.evaluation import evaluate_generation
from src.training.protocol import cohort_records, evaluation_context, freeze_protocol, consume_protocol
from src.training.statistics import bootstrap_interval, feature_distribution_metrics
from src.utils.integrity import sha256_file
from src.utils.artifacts import reference, resolve_reference


def processed(root, prefix, count=4):
    root.mkdir()
    rows = []
    for i in range(count):
        array = np.random.default_rng(i+sum(map(ord, prefix))).uniform(-1, 1, (1, 8, 8, 8)).astype(np.float32)
        path = root/f'{i}.npy'
        np.save(path, array)
        rows.append(dict(scan_id=f'{prefix}-{i}', patient_id=f'{prefix}-patient-{i}',
                         split='train' if i < count-2 else 'validation' if i == count-2 else 'test',
                         output_file=path.name, output_sha256=sha256_file(path), source_sha256=f'{prefix}-source-{i}'))
    with (root/'metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def test_external_evaluation_separates_lineage_and_checks_model_selection_overlap(tmp_path):
    train, external = tmp_path/'train', tmp_path/'external'
    rows = processed(train, 'training')
    processed(external, 'external')
    checkpoint = dict(dataset_fingerprint=fingerprint(train), cohort_records=cohort_records(train))
    context = evaluation_context(external, checkpoint, 'test')
    assert context['independent_patient_test_eligible']
    assert context['training_dataset_fingerprint'] != context['evaluation_dataset_fingerprint']
    with pytest.raises(ValueError, match='legacy checkpoint'):
        evaluation_context(external, dict(dataset_fingerprint=fingerprint(train)), 'test')
    assert evaluation_context(external, dict(dataset_fingerprint=fingerprint(train)), 'test', train)['same_dataset'] is False
    checkpoint['cohort_records'][2]['patient_id'] = 'external-patient-3'
    with pytest.raises(ValueError, match='overlaps'):
        evaluation_context(external, checkpoint, 'test')
    assert rows[2]['split'] == 'validation'  # model-selection cases also count as previously seen


def test_frozen_protocol_and_single_completed_test_receipt(tmp_path):
    train, external = tmp_path/'train', tmp_path/'external'
    processed(train, 'training'); processed(external, 'external')
    path = tmp_path/'plan.json'
    plan = freeze_protocol(train, external, path)
    assert plan['independent_patient_test_eligible']
    with pytest.raises(ValueError, match='already exists'):
        freeze_protocol(train, external, path)
    context = evaluation_context(external, dict(dataset_fingerprint=fingerprint(train)), 'test', train)
    consume_protocol(path, context, 'test', tmp_path/'results')
    with pytest.raises(FileExistsError):
        consume_protocol(path, context, 'test', tmp_path/'second-results')
    with pytest.raises(ValueError, match='frozen protocol'):
        consume_protocol(path, context, 'validation', tmp_path/'wrong-results')


def test_missing_patient_ids_never_becomes_patient_disjoint_claim(tmp_path):
    train, external = tmp_path/'train', tmp_path/'external'
    processed(train, 'training'); processed(external, 'external')
    records = cohort_records(train)
    for row in records:
        row['patient_id'] = ''
    result = evaluation_context(external, dict(dataset_fingerprint=fingerprint(train), cohort_records=records), 'test')
    assert not result['independent_patient_test_eligible']
    assert not result['overlap_audit']['patient_identifiers_available']


def test_memorization_search_includes_training_examples_beyond_diversity_limit(tmp_path):
    root, generated = tmp_path/'real', tmp_path/'generated'
    rows = processed(root, 'cohort', 12)
    generated.mkdir()
    samples = []
    for i in (8, 9):
        source = root/f'{i}.npy'
        target = generated/f'{i}.npy'
        target.write_bytes(source.read_bytes())
        samples.append(dict(file=target.name, sha256=sha256_file(target)))
    manifest = dict(rows=samples, shape=[8, 8, 8], generator_train_ids=[r['scan_id'] for r in rows if r['split']=='train'],
                    dataset_fingerprint=fingerprint(root), smoke_test=True)
    (generated/'manifest.json').write_text(json.dumps(manifest))
    report = evaluate_generation(generated, root, tmp_path/'report.json', max_samples=2)
    assert report['real_count'] == 2 and report['memorization_reference_count'] == 10
    assert report['nearest_training_mse'] == [0., 0.]
    assert [r['scan_id'] for r in report['nearest_training_neighbors']] == ['cohort-8', 'cohort-9']
    assert [r['mse'] for r in report['nearest_training_spatial_features']] == [0., 0.]


def test_bootstrap_uses_patient_clusters_and_repeats_exactly():
    values = [0., 0., 1., 1.]
    result = bootstrap_interval(values, groups=['a', 'a', 'b', 'b'], resamples=500)
    assert result['clusters'] == 2 and result['observations'] == 4
    assert result['mean'] == .5 and result['lower'] == 0 and result['upper'] == 1
    assert result == bootstrap_interval(values, groups=['a', 'a', 'b', 'b'], resamples=500)
    assert not bootstrap_interval([1.])['uncertainty_available']
    with pytest.raises(ValueError):
        bootstrap_interval([float('nan')])


def test_feature_metrics_match_closed_form_reference():
    result = feature_distribution_metrics([[0.], [1.]], [[2.], [3.]])
    assert result['frechet_feature_distance'] == pytest.approx(4.)
    assert result['polynomial_kid_unbiased'] == pytest.approx(297.5)
    identical = feature_distribution_metrics([[0., 1.], [1., 0.], [2., 1.]], [[0., 1.], [1., 0.], [2., 1.]])
    assert identical['frechet_feature_distance'] == pytest.approx(0., abs=1e-12)


def test_relative_artifact_reference_survives_a_moved_project(tmp_path):
    project = tmp_path/'project'
    owner = project/'runs'/'example'
    data = project/'data'
    owner.mkdir(parents=True); data.mkdir()
    relative = reference(data, owner)
    moved = tmp_path/'moved'
    project.rename(moved)
    assert resolve_reference(moved/'runs/example', relative) == moved/'data'


def test_external_reconstruction_executes_a_frozen_protocol_end_to_end(tmp_path, monkeypatch):
    from src.models.vqvae import VQVAE3D
    from src.training.common import save_checkpoint
    from src.training.experiments import evaluate_reconstruction
    train, external = tmp_path/'train', tmp_path/'external'
    training_rows = processed(train, 'training'); processed(external, 'external')
    # Untrained numerical fixture checkpoint; never a research-result artifact.
    model = VQVAE3D(base=4, latent_channels=4, codes=8)
    checkpoint = tmp_path/'test-only-vq.pt'
    save_checkpoint(checkpoint, dict(kind='vq', model=model.state_dict(), model_config=model.config,
        shape=[8, 8, 8], smoke_test=True, dataset_fingerprint=fingerprint(train),
        cohort_records=cohort_records(train), train_ids=[r['scan_id'] for r in training_rows if r['split']=='train']))
    plan = tmp_path/'protocol.json'; freeze_protocol(train, external, plan)
    output = tmp_path/'evaluation'
    report = evaluate_reconstruction(external, checkpoint, output, split='test', device='cpu',
                                     training_root=train, protocol=plan)
    assert report['smoke_test'] and len(report['rows']) == 1
    assert report['evaluation_context']['evaluation_dataset_fingerprint'] == fingerprint(external)
    assert report['evaluation_context']['training_dataset_fingerprint'] == fingerprint(train)
    assert report['frozen_protocol']['independent_patient_test_eligible']
    assert resolve_reference(output, report['input_dataset']) == external
    receipt = json.loads(plan.with_suffix('.evaluation.json').read_text())
    assert receipt['status'] == 'completed'
    assert receipt['report_sha256'] == sha256_file(output/'report.json')
    assert receipt['checkpoint_sha256'] == sha256_file(checkpoint)
    def unexpected_inference(*args, **kwargs):
        pytest.fail('Rejected protocol reached model inference')
    monkeypatch.setattr(VQVAE3D, 'forward', unexpected_inference)
    with pytest.raises(FileExistsError):
        evaluate_reconstruction(external, checkpoint, tmp_path/'repeat-evaluation', split='test', device='cpu',
                                 training_root=train, protocol=plan)
    assert not (tmp_path/'repeat-evaluation').exists()
    with pytest.raises(ValueError, match='frozen protocol'):
        evaluate_reconstruction(external, checkpoint, tmp_path/'wrong-split', split='validation', device='cpu',
                                 training_root=train, protocol=plan)
    assert not (tmp_path/'wrong-split').exists()


def test_content_overlap_is_detected_without_declared_hashes(tmp_path):
    train, external = tmp_path/'train', tmp_path/'external'
    training_rows = processed(train, 'training')
    rows = processed(external, 'external')
    (external/'3.npy').write_bytes((train/'0.npy').read_bytes())
    rows[3]['output_sha256'] = rows[3]['source_sha256'] = ''
    with (external/'metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    before = (external/'metadata.csv').read_bytes()
    checkpoint = dict(dataset_fingerprint=fingerprint(train), cohort_records=cohort_records(train))
    for training_reference in (None, train):
        with pytest.raises(ValueError, match='overlaps'):
            evaluation_context(external, checkpoint, 'test', training_reference)
    with pytest.raises(ValueError, match='overlapping'):
        freeze_protocol(train, external, tmp_path/'invalid-plan.json')
    assert cohort_records(external)[3]['output_sha256'] == training_rows[0]['output_sha256']
    assert (external/'metadata.csv').read_bytes() == before


def test_incomplete_embedded_hashes_require_verified_training_root(tmp_path):
    train, external = tmp_path/'train', tmp_path/'external'
    processed(train, 'training'); processed(external, 'external')
    records = cohort_records(train)
    records[0]['output_sha256'] = ''
    checkpoint = dict(dataset_fingerprint=fingerprint(train), cohort_records=records)
    with pytest.raises(ValueError, match='training_root'):
        evaluation_context(external, checkpoint, 'test')
    result = evaluation_context(external, checkpoint, 'test', train)
    assert result['overlap_audit']['tensor_hashes_complete']
    assert result['independent_patient_test_eligible']


def test_verified_snapshot_preserves_legacy_fingerprint_and_checks_declared_hashes(tmp_path):
    import hashlib
    from src.utils.cohort import verify_cohort
    root = tmp_path/'data'; rows = processed(root, 'cohort')
    digest = hashlib.sha256((root/'metadata.csv').read_bytes())
    for row in rows:
        digest.update(row['output_file'].encode()); digest.update(row['output_sha256'].encode())
    assert verify_cohort(root).fingerprint == digest.hexdigest()
    (root/'0.npy').write_bytes((root/'1.npy').read_bytes())
    with pytest.raises(ValueError, match='checksum mismatch'):
        cohort_records(root)


def test_protocol_reservation_is_exclusive_and_retains_failures(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from src.training.protocol import reserve_protocol, evaluation_attempt
    train, external = tmp_path/'train', tmp_path/'external'
    processed(train, 'training'); processed(external, 'external')
    plan = tmp_path/'plan.json'; freeze_protocol(train, external, plan)
    context = evaluation_context(external, dict(dataset_fingerprint=fingerprint(train)), 'test', train)
    def reserve():
        try:
            return reserve_protocol(plan, context, 'test', tmp_path/'results', 'checkpoint-hash')
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reserve(), range(2)))
    assert sum(r is not None for r in results) == 1
    assert json.loads(plan.with_suffix('.evaluation.json').read_text())['status'] == 'running'
    failed = tmp_path/'failed.json'; freeze_protocol(train, external, failed)
    with pytest.raises(RuntimeError, match='injected'):
        with evaluation_attempt(failed, context, 'test', tmp_path/'failed-output', 'checkpoint-hash'):
            raise RuntimeError('injected inference failure')
    receipt = json.loads(failed.with_suffix('.evaluation.json').read_text())
    assert receipt['status'] == 'failed' and receipt['error_type'] == 'RuntimeError'
    with pytest.raises(FileExistsError):
        reserve_protocol(failed, context, 'test', tmp_path/'retry')


def test_report_failure_is_recorded_and_legacy_receipts_block_reuse(tmp_path, monkeypatch):
    from src.training.protocol import evaluation_attempt, reserve_protocol
    train, external = tmp_path/'train', tmp_path/'external'
    processed(train, 'training'); processed(external, 'external')
    plan = tmp_path/'plan.json'; freeze_protocol(train, external, plan)
    context = evaluation_context(external, dict(dataset_fingerprint=fingerprint(train)), 'test', train)
    with pytest.raises(FileNotFoundError):
        with evaluation_attempt(plan, context, 'test', tmp_path/'absent-report', 'checkpoint-hash'):
            pass
    assert json.loads(plan.with_suffix('.evaluation.json').read_text())['status'] == 'failed'
    legacy = tmp_path/'legacy.json'; freeze_protocol(train, external, legacy)
    legacy.with_suffix('.evaluation.json').write_text('{"output": "legacy-completed"}')
    with pytest.raises(FileExistsError):
        reserve_protocol(legacy, context, 'test', tmp_path/'legacy-retry')
