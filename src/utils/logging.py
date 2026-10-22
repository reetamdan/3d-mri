import csv
import json
from pathlib import Path
import os
import uuid


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def atomic_json(path: str | Path, value) -> None:
    """Publish a complete JSON document, retaining the old file on failure."""
    path = Path(path)
    partial = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.partial')
    try:
        with partial.open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def write_csv(path: str | Path, rows: list[dict], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
