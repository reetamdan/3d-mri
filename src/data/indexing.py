"""Index de-identified subjects. Never infer subjects from flat filenames."""
import csv
import hashlib
from pathlib import Path


def token(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def index_dataset(config: dict) -> list[dict]:
    root = Path(config['raw_dir']).resolve()
    if not root.is_dir():
        raise ValueError('Raw dataset directory does not exist')
    records = []
    if config.get('manifest'):
        with Path(config['manifest']).open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not {'file_path', 'patient_id'}.issubset(reader.fieldnames or []):
                raise ValueError('Manifest requires file_path and patient_id columns')
            for row in reader:
                path = (root / row['file_path']).resolve()
                if not path.is_relative_to(root) or not row['patient_id'].strip():
                    raise ValueError('Manifest paths must stay inside raw_dir and patient_id must be nonempty')
                records.append((path, row['patient_id'].strip()))
    else:
        paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name.lower().endswith(('.nii', '.nii.gz')))
        for path in paths:
            relative = path.relative_to(root)
            if len(relative.parts) < 2:
                raise ValueError('Flat datasets require a manifest with subject identifiers to prevent split leakage')
            records.append((path, relative.parts[0]))
    seen = set()
    result = []
    for path, patient in records:
        if path in seen:
            raise ValueError('Duplicate file in dataset manifest')
        seen.add(path)
        result.append({'scan_id': token(path.relative_to(root).as_posix()), 'patient_id': token(patient), 'path': str(path)})
    return sorted(result, key=lambda r: r['scan_id'])
