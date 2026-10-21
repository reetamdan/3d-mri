"""Separate training lineage from evaluation identity; audit overlap explicitly."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from src.utils.logging import write_json
from src.utils.integrity import sha256_file
from .common import fingerprint

IDENTITIES = ('scan_id', 'patient_id', 'exam_id', 'source_sha256', 'output_sha256')


def cohort_records(root):
    with (Path(root)/'metadata.csv').open(newline='', encoding='utf-8') as f:
        return [{key: row.get(key, '') for key in (*IDENTITIES, 'split', 'labels', 'plane')}
                for row in csv.DictReader(f)]


def audit_overlap(reference_rows, evaluation_rows):
    if not evaluation_rows:
        raise ValueError('Evaluation cohort is empty')
    overlaps = {}
    for key in IDENTITIES:
        known = {r.get(key) for r in reference_rows if r.get(key)}
        shared = known & {r.get(key) for r in evaluation_rows if r.get(key)}
        if shared:
            overlaps[key] = sorted(shared)
    patients = all(r.get('patient_id') for r in [*reference_rows, *evaluation_rows])
    return dict(overlap_counts={k: len(v) for k, v in overlaps.items()},
                overlap_examples={k: v[:5] for k, v in overlaps.items()},
                patient_identifiers_available=patients,
                patient_disjoint_verified=patients and not overlaps,
                note='Disjoint scan/exam IDs do not establish patient disjointness when patient IDs are absent.')


def evaluation_context(root, checkpoint, split, training_root=None):
    evaluation_fingerprint = fingerprint(root)
    same = evaluation_fingerprint == checkpoint['dataset_fingerprint']
    if not same and training_root is None and not checkpoint.get('cohort_records'):
        raise ValueError('External evaluation of a legacy checkpoint requires its training_root for overlap checks')
    if training_root is not None:
        if fingerprint(training_root) != checkpoint['dataset_fingerprint']:
            raise ValueError('Training reference does not match checkpoint lineage')
        records = cohort_records(training_root)
    else:
        records = checkpoint.get('cohort_records') or cohort_records(root)
    selected = [r for r in cohort_records(root) if r['split'] == split]
    seen_splits = ('train',) if same and split == 'validation' else ('train', 'validation')
    seen = [r for r in records if r['split'] in seen_splits]
    audit = audit_overlap(seen, selected)
    if split == 'train' and same:
        role = 'training reconstruction; not held out'
    else:
        if audit['overlap_counts']:
            raise ValueError('Evaluation cohort overlaps training/model-selection identities or source/tensor contents')
        role = 'model-selection validation' if same and split == 'validation' else 'external/test evaluation'
    return dict(training_dataset_fingerprint=checkpoint['dataset_fingerprint'],
                evaluation_dataset_fingerprint=evaluation_fingerprint, same_dataset=same,
                evaluation_role=role, overlap_audit=audit,
                independent_patient_test_eligible=split == 'test' and audit['patient_disjoint_verified'])


def freeze_protocol(training_root, evaluation_root, output, split='test', seeds=(42, 43, 44),
                    primary_metric='ssim', hypothesis='Compare reconstruction and generation baselines'):
    output = Path(output)
    if output.exists():
        raise ValueError('Protocol already exists; use a new protocol instead of overwriting the plan')
    records = cohort_records(training_root)
    selected = [r for r in cohort_records(evaluation_root) if r['split'] == split]
    audit = audit_overlap([r for r in records if r['split'] in ('train', 'validation')], selected)
    if audit['overlap_counts']:
        raise ValueError('Cannot freeze an overlapping test cohort')
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Declare distinct experiment seeds')
    protocol = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        training_fingerprint=fingerprint(training_root),
        evaluation_fingerprint=fingerprint(evaluation_root), evaluation_split=split,
        evaluation_ids=[r['scan_id'] for r in selected], seeds=list(seeds), primary_metric=primary_metric,
        hypothesis=hypothesis, overlap_audit=audit,
        independent_patient_test_eligible=split == 'test' and audit['patient_disjoint_verified'],
        expert_review='Not performed by software; requires completed review records',
        note='Freeze before test inspection. This file records a plan, not proof of prospective compliance.')
    write_json(output, protocol)
    return protocol


def consume_protocol(path, context, split, output):
    """Record one completed test evaluation per frozen plan; cannot relabel validation."""
    path = Path(path)
    protocol = json.loads(path.read_text())
    if (protocol['training_fingerprint'] != context['training_dataset_fingerprint']
            or protocol['evaluation_fingerprint'] != context['evaluation_dataset_fingerprint']
            or protocol['evaluation_split'] != split):
        raise ValueError('Evaluation does not match the frozen protocol')
    receipt = path.with_suffix('.evaluation.json')
    payload = dict(protocol_sha256=sha256_file(path), output=str(Path(output).resolve()),
                   independent_patient_test_eligible=protocol['independent_patient_test_eligible'])
    # Exclusive creation prevents silently reusing a frozen test plan.
    with receipt.open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return payload
