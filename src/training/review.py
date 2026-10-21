"""Blinded volume-review packet preparation. Scores must come from actual reviewers."""
import csv
import json
from pathlib import Path
import numpy as np
from src.utils.integrity import generated_volume, sha256_file
from src.utils.logging import write_json, write_csv
from .common import TrainingDataset, fresh_output
from .evidence import slice_grid

SCORES = ('appearance', 'anatomy', 'continuity')


def prepare_review(real_root, generated_dir, output, count=8, seed=42, split='validation'):
    if count < 1:
        raise ValueError('Review count must be positive')
    generated_dir = Path(generated_dir)
    manifest = json.loads((generated_dir/'manifest.json').read_text())
    real = TrainingDataset(real_root, split, manifest['shape'])
    if min(len(real), len(manifest['rows'])) < count:
        raise ValueError('Review needs the requested number of real and generated cases')
    rng = np.random.default_rng(seed)
    selected_real = rng.permutation(len(real))[:count]
    selected_generated = rng.permutation(len(manifest['rows']))[:count]
    cases = [('real', int(i)) for i in selected_real]+[('generated', int(i)) for i in selected_generated]
    rng.shuffle(cases)
    root = fresh_output(output)
    (root/'.gitignore').write_text('*\n', encoding='utf-8')
    (root/'cases').mkdir()
    key, rows = [], []
    for i, (kind, index) in enumerate(cases):
        case = f'case-{i+1:03d}'
        if kind == 'real':
            volume = real[index][0].numpy()
            identity = real.rows[index]['scan_id']
        else:
            record = manifest['rows'][index]
            volume = generated_volume(generated_dir, record, manifest['shape'])
            identity = record['file']
        np.save(root/'cases'/f'{case}.npy', volume, allow_pickle=False)
        slice_grid([volume[0]], [case], root/'cases'/f'{case}.png')
        key.append(dict(case_id=case, origin=kind, source_identity=identity,
                        sha256=sha256_file(root/'cases'/f'{case}.npy')))
        rows.append(dict(reviewer_id='', case_id=case, appearance='', anatomy='', continuity='',
                         suspected_synthetic='', comments=''))
    write_csv(root/'review-template.csv', rows, list(rows[0]))
    write_json(root/'unblinding-key.json', dict(seed=seed, cases=key, generated_manifest_sha256=sha256_file(generated_dir/'manifest.json')))
    (root/'REVIEW_INSTRUCTIONS.md').write_text(
        '# Blinded research review\n\nShare only cases/, this document, and the blank CSV with reviewers. '
        'Keep unblinding-key.json separate until scores are locked. Use at least two qualified reviewers '
        'and record qualifications independently. Do not infer expertise from a software-generated score.\n\n'
        'Score appearance, anatomical plausibility and slice continuity from 1 (poor) to 5 (strong). '
        'Record suspected_synthetic as 0 or 1. Review full NPY tensors [1,row,column,slice] before scoring '
        'continuity; PNGs alone are insufficient. There is no patient-space affine. '
        'These scores describe this packet, not diagnostic utility. '
        'Predeclare inclusion criteria and instructions before review; preserve disagreements and failures.\n', encoding='utf-8')
    return dict(cases=len(cases), count_per_origin=count, scores_collected=False,
                note='Packet prepared only; no expert review or anatomical validation has occurred')


def summarize_review(packet, scores_csv, output):
    packet = Path(packet)
    key = {r['case_id']: r['origin'] for r in json.loads((packet/'unblinding-key.json').read_text())['cases']}
    with Path(scores_csv).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    seen, reviewers = set(), set()
    for row in rows:
        identity = (row.get('reviewer_id', '').strip(), row.get('case_id'))
        if not identity[0] or identity[1] not in key or identity in seen:
            raise ValueError('Review rows require actual reviewer IDs, known cases, and no duplicate ratings')
        seen.add(identity); reviewers.add(identity[0])
        if any(row.get(metric) not in ('1', '2', '3', '4', '5') for metric in SCORES):
            raise ValueError('Actual complete 1..5 scores are required; blank templates are not results')
        if row.get('suspected_synthetic') not in ('0', '1'):
            raise ValueError('suspected_synthetic must be 0 or 1')
    if len(reviewers) < 2 or any((reviewer, case) not in seen for reviewer in reviewers for case in key):
        raise ValueError('Require >=2 reviewers and complete ratings for every case')
    summary = {}
    for origin in ('real', 'generated'):
        selected = [r for r in rows if key[r['case_id']] == origin]
        summary[origin] = {metric: dict(mean=float(np.mean([int(r[metric]) for r in selected])),
                                        ratings=len(selected)) for metric in SCORES}
    report = dict(reviewers=sorted(reviewers), cases=len(key), origin_scores=summary,
                  scores_sha256=sha256_file(scores_csv), qualifications_verified_by_software=False,
                  interpretation='Descriptive completed ratings; no clinical utility or inter-rater reliability claim is automated')
    write_json(output, report)
    return report
