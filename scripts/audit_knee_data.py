"""Array-grid, intensity and source-shape diagnostics without inventing MRI geometry."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
from src.training.common import fresh_output
from src.utils.cohort import verify_cohort
from src.utils.integrity import confined_path
from src.utils.artifacts import resolve_reference
from src.utils.logging import write_json
from src.training.evidence import slice_grid


def histogram_image(counts, title, path):
    counts = np.asarray(counts, dtype=float)
    image = Image.new('RGB', (900, 320), '#0d141c'); draw = ImageDraw.Draw(image)
    draw.text((20, 15), title, fill='white')
    scale = max(float(counts.max()), 1.)
    for i, count in enumerate(counts):
        x = 35+i*830/len(counts)
        draw.rectangle((x, 280-count/scale*230, x+max(1, 830/len(counts)-1), 280), fill='#58cbbb')
    image.save(path)


def audit(data, output, seed=42):
    data, output = Path(data), fresh_output(output)
    snapshot = verify_cohort(data)
    experiment = json.loads((data/'experiment.json').read_text())
    rows: list[dict[str, Any]] = []
    intensity_counts = np.zeros(64, dtype=np.int64)
    for row in snapshot.rows:
        array = np.load(confined_path(data, row['output_file']), allow_pickle=False)
        if array.ndim != 4 or array.shape[0] != 1 or not np.isfinite(array).all():
            raise ValueError('Invalid tensor layout/intensities')
        x = array[0]
        intensity_counts += np.histogram(x, bins=64, range=(-1, 1))[0]
        rows.append(dict(scan_id=row['scan_id'], split=row['split'], shape=list(x.shape),
                         original_shape=json.loads(row['original_shape']) if row.get('original_shape') else None,
                         minimum=float(x.min()), maximum=float(x.max()), mean=float(x.mean()), std=float(x.std()),
                         constant_slices=int((np.ptp(x, axis=(0, 1)) < 1e-6).sum()),
                         near_background_slices=int(((x.std((0, 1)) < .01) & (x.mean((0, 1)) < -.98)).sum())))
    raw_depths = [int(r['original_shape'][0]) for r in rows if r['original_shape']]
    source = resolve_reference(data, experiment['root']) if experiment.get('root') else None
    source_map = {}
    if source is not None and (source/'source_manifest.json').is_file():
        source_map = {r['sha256']: confined_path(source, r['file'])
                      for r in json.loads((source/'source_manifest.json').read_text())['files']}
    choices = np.random.default_rng(seed).permutation(len(rows))[:6]
    for rank, i in enumerate(choices):
        record = snapshot.rows[int(i)]
        processed = np.load(confined_path(data, record['output_file']), allow_pickle=False)[0]
        volumes, labels = [processed], ['Processed; acquired-series fractions']
        raw_path = source_map.get(record.get('source_sha256'))
        if raw_path:
            raw = np.load(raw_path, allow_pickle=False).transpose(1, 2, 0).astype(np.float32)
            raw = (raw-raw.min())/(raw.max()-raw.min())*2-1
            volumes.insert(0, raw); labels.insert(0, 'Source; acquired-series fractions')
        slice_grid(volumes, labels, output/f'random-case-{rank+1}.png')
        cross = []
        for axis in range(3):
            plane = np.take(processed, processed.shape[axis]//2, axis=axis)
            cross.append(np.repeat(plane[..., None], 4, axis=-1))
        slice_grid(cross, ['Central row cross-section', 'Central column cross-section', 'Central acquired slice'],
                   output/f'central-axes-{rank+1}.png')
    histogram_image(intensity_counts, 'Normalized intensity histogram: [-1, 1], 64 bins', output/'intensity-histogram.png')
    if raw_depths:
        bins = list(range(min(raw_depths), max(raw_depths)+2))
        histogram_image(np.histogram(raw_depths, bins=bins)[0], f'Original slice count: {min(raw_depths)} to {max(raw_depths)}',
                        output/'slice-count-histogram.png')
    report = dict(dataset_fingerprint=snapshot.fingerprint, rows=rows, seed=seed,
        selected_visual_cases=[rows[int(i)]['scan_id'] for i in choices],
        split_counts=dict(Counter(r['split'] for r in rows)),
        output_shapes=dict(Counter(str(r['shape']) for r in rows)),
        original_shapes=dict(Counter(str(r['original_shape']) for r in rows)),
        original_slice_range=[min(raw_depths), max(raw_depths)] if raw_depths else None,
        intensity_histogram=dict(counts=intensity_counts.tolist(), range=[-1, 1]),
        normalization='Volume-wise min-max after trilinear resizing, [-1,1]',
        tensor_layout='[C,row,column,acquired-series slice]', acquired_plane=experiment.get('plane'),
        spacing_available=False, physical_orientation_available=False, laterality_available=False,
        physical_spacing_histogram=None, patient_linkage_available=all(bool(r.get('patient_id')) for r in snapshot.rows),
        explicit_crop=False, explicit_padding_fraction=0.,
        limitation='MRNet source arrays cannot establish patient-space orientation, true laterality, mm spacing or scanner/sequence consistency; no automatic flips are applied.')
    write_json(output/'data-audit.json', report)
    print(json.dumps({k: report[k] for k in ('split_counts', 'output_shapes', 'original_slice_range', 'limitation')}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/mrnet-m4-20261021')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.data, args.output)


if __name__ == '__main__':
    main()
