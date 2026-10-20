from collections import Counter
import numpy as np


def summarize(records: list[dict], errors: list[dict], discovered: int) -> dict:
    def stats(key):
        if not records:
            return {'min': None, 'max': None, 'mean': None}
        x = np.asarray([r[key] for r in records], dtype=float)
        return {'min': x.min(axis=0).tolist(), 'max': x.max(axis=0).tolist(), 'mean': x.mean(axis=0).tolist()}
    return {
        'discovered_scans': discovered, 'valid_scans': len(records), 'invalid_scans': len(errors),
        'subjects': len({r['patient_id'] for r in records}),
        'shape': stats('volume_shape'), 'spacing_native_units': stats('voxel_spacing'),
        'spatial_units': dict(Counter(r.get('spatial_unit', 'unknown') for r in records)),
        'intensity_minimum': stats('minimum_intensity'), 'intensity_maximum': stats('maximum_intensity'),
        'per_scan_intensity_mean': stats('mean_intensity'), 'per_scan_intensity_std': stats('standard_deviation'),
        'distinct_shapes': len({tuple(r['volume_shape']) for r in records}),
        'errors_by_category': dict(Counter(r['code'] for r in errors)),
        'note': 'Intensity means/std are per-scan summaries. Spacing aggregates are comparable only when units match. Missing files are detectable through a manifest.',
    }
