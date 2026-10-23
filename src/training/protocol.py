"""Separate training lineage from evaluation identity; audit overlap explicitly."""
import json
import hashlib
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from src.utils.logging import write_json, atomic_json
from src.utils.integrity import sha256_file
from src.utils.cohort import IDENTITIES, verify_cohort, has_tensor_hashes
from src.utils.artifacts import reference
from typing import Any, Iterator


def cohort_records(root: str | Path) -> list[dict[str, str]]:
    return verify_cohort(root).records()


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
    content_complete = has_tensor_hashes(reference_rows) and has_tensor_hashes(evaluation_rows)
    return dict(overlap_counts={k: len(v) for k, v in overlaps.items()},
                overlap_examples={k: v[:5] for k, v in overlaps.items()},
                patient_identifiers_available=patients,
                tensor_hashes_complete=content_complete,
                identity_coverage={key: dict(reference=sum(bool(r.get(key)) for r in reference_rows),
                                             evaluation=sum(bool(r.get(key)) for r in evaluation_rows))
                                   for key in IDENTITIES},
                reference_count=len(reference_rows), evaluation_count=len(evaluation_rows),
                patient_disjoint_verified=patients and content_complete and not overlaps,
                content_check='Processed-file SHA-256 verified; source hashes compared when supplied',
                note='Disjoint scan/exam IDs do not establish patient disjointness when patient IDs are absent.')


def evaluation_context(root, checkpoint, split, training_root=None):
    evaluation = verify_cohort(root)
    evaluation_fingerprint = evaluation.fingerprint
    same = evaluation_fingerprint == checkpoint['dataset_fingerprint']
    if not same and training_root is None and not checkpoint.get('cohort_records'):
        raise ValueError('External evaluation of a legacy checkpoint requires its training_root for overlap checks')
    if training_root is not None:
        training = evaluation if Path(training_root).resolve() == Path(root).resolve() else verify_cohort(training_root)
        if training.fingerprint != checkpoint['dataset_fingerprint']:
            raise ValueError('Training reference does not match checkpoint lineage')
        records = training.records()
    else:
        records = evaluation.records() if same else checkpoint['cohort_records']
    selected = [r for r in evaluation.records() if r['split'] == split]
    seen_splits = ('train',) if same and split == 'validation' else ('train', 'validation')
    seen = [r for r in records if r['split'] in seen_splits]
    # Representation pretraining is also prior exposure, even when fine-tuning
    # uses the same local cohort fingerprint and model-selection split.
    seen += checkpoint.get('representation_pretraining_records', [])
    if not same and not has_tensor_hashes(seen):
        raise ValueError('External evaluation of a legacy checkpoint with incomplete tensor hashes requires training_root')
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
                 evaluation_ids=[r['scan_id'] for r in selected],
                independent_patient_test_eligible=split == 'test' and audit['patient_disjoint_verified'])


def freeze_protocol(training_root, evaluation_root, output, split='test', seeds=(42, 43, 44),
                    primary_metric='ssim', hypothesis='Compare reconstruction and generation baselines'):
    output = Path(output)
    if output.exists():
        raise ValueError('Protocol already exists; use a new protocol instead of overwriting the plan')
    training, evaluation = verify_cohort(training_root), verify_cohort(evaluation_root)
    records = training.records()
    selected = [r for r in evaluation.records() if r['split'] == split]
    audit = audit_overlap([r for r in records if r['split'] in ('train', 'validation')], selected)
    if audit['overlap_counts']:
        raise ValueError('Cannot freeze an overlapping test cohort')
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Declare distinct experiment seeds')
    protocol = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        training_fingerprint=training.fingerprint,
        evaluation_fingerprint=evaluation.fingerprint, evaluation_split=split,
        evaluation_ids=[r['scan_id'] for r in selected], seeds=list(seeds), primary_metric=primary_metric,
        hypothesis=hypothesis, overlap_audit=audit,
        independent_patient_test_eligible=split == 'test' and audit['patient_disjoint_verified'],
        expert_review='Not performed by software; requires completed review records',
        note='Freeze before test inspection. This file records a plan, not proof of prospective compliance.')
    write_json(output, protocol)
    return protocol


@dataclass
class ProtocolAttempt:
    receipt: Path
    payload: dict[str, Any]

    def reference(self) -> dict[str, Any]:
        return {key: self.payload[key] for key in ('attempt_id', 'protocol_sha256', 'checkpoint_sha256',
                                                  'output', 'independent_patient_test_eligible')}

    def finish(self, status: str, **details: Any) -> None:
        self.payload.update(status=status, finished_at=datetime.now(timezone.utc).isoformat(), **details)
        atomic_json(self.receipt, self.payload)


def reserve_protocol(path, context, split, output, checkpoint_sha256=None) -> ProtocolAttempt:
    """Exclusively reserve a validated plan before any model inference/output writes.

    Any existing receipt (including legacy, failed, or interrupted attempts) blocks
    automatic reuse. An interrupted attempt stays visible for explicit recovery.
    """
    path = Path(path)
    encoded = path.read_bytes()
    protocol = json.loads(encoded)
    if (protocol['training_fingerprint'] != context['training_dataset_fingerprint']
            or protocol['evaluation_fingerprint'] != context['evaluation_dataset_fingerprint']
            or protocol['evaluation_split'] != split
            or protocol['evaluation_ids'] != context['evaluation_ids']):
        raise ValueError('Evaluation does not match the frozen protocol')
    receipt = path.with_suffix('.evaluation.json')
    payload = dict(schema_version=2, attempt_id=uuid.uuid4().hex, status='running',
                   started_at=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=hashlib.sha256(encoded).hexdigest(), checkpoint_sha256=checkpoint_sha256,
                   output=str(Path(output).resolve()),
                   independent_patient_test_eligible=context['independent_patient_test_eligible'])
    with receipt.open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return ProtocolAttempt(receipt, payload)


@contextmanager
def evaluation_attempt(path, context, split, output, checkpoint_sha256) -> Iterator[dict[str, Any] | None]:
    attempt = reserve_protocol(path, context, split, output, checkpoint_sha256) if path else None
    try:
        record = attempt.reference() if attempt else None
        if record is not None and attempt is not None:
            record['receipt_file'] = reference(attempt.receipt, output)
        yield record
        if attempt:
            attempt.finish('completed', report_sha256=sha256_file(Path(output)/'report.json'))
    except BaseException as exc:
        if attempt:
            attempt.finish('failed', error_type=type(exc).__name__)
        raise


def consume_protocol(path, context, split, output):
    """Compatibility helper for clients recording an already-completed evaluation.

    Inference entry points must use evaluation_attempt before running the model.
    """
    attempt = reserve_protocol(path, context, split, output)
    attempt.finish('completed')
    return attempt.reference()
