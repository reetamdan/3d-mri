"""One content-verified dataset snapshot shared by fingerprints and overlap audits."""
import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Callable

from .integrity import confined_path, sha256_file

IDENTITIES = ('scan_id', 'patient_id', 'exam_id', 'source_sha256', 'output_sha256')
COHORT_FIELDS = (*IDENTITIES, 'split', 'labels', 'plane')


@dataclass(frozen=True)
class VerifiedCohort:
    fingerprint: str
    rows: list[dict[str, str]]

    def records(self) -> list[dict[str, str]]:
        return [{key: row.get(key, '') for key in COHORT_FIELDS} for row in self.rows]


def verify_cohort(root: str | Path, hash_file: Callable = sha256_file) -> VerifiedCohort:
    """Derive tensor hashes without rewriting metadata or changing legacy fingerprints.

    Core callers use uncached SHA-256. The dashboard may supply its signature-keyed
    file hasher; that optimization is separate from research evaluation.
    """
    root = Path(root).resolve()
    metadata = (root / 'metadata.csv').read_bytes()
    digest = hashlib.sha256(metadata)
    rows = []
    for raw in csv.DictReader(io.StringIO(metadata.decode('utf-8'), newline='')):
        row = dict(raw)
        actual = hash_file(confined_path(root, row['output_file']))
        if row.get('output_sha256') and actual != row['output_sha256']:
            raise ValueError('Processed tensor checksum mismatch; reprocess the source dataset')
        digest.update(row['output_file'].encode())
        digest.update(actual.encode())
        row['output_sha256'] = actual
        rows.append(row)
    return VerifiedCohort(digest.hexdigest(), rows)


def has_tensor_hashes(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(len(r.get('output_sha256', '')) == 64
                              and all(c in '0123456789abcdef' for c in r['output_sha256']) for r in rows)
