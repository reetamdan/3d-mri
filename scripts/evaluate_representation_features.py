"""Frozen medical descriptors and stronger memorization screens on existing outputs."""
import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F
from src.metrics import reconstruction_metrics
from src.training.common import TrainingDataset, device_for, fresh_output
from src.training.evidence import slice_grid
from src.training.statistics import feature_distribution_metrics
from src.utils.cohort import verify_cohort
from src.utils.integrity import generated_volume, sha256_file
from src.utils.logging import write_json


def embeddings(model, arrays, device):
    with torch.no_grad():
        return np.stack([model(torch.from_numpy(x.copy())[None].to(device)).cpu().numpy()[0] for x in arrays])


def cosine_distances(first, second):
    a = first/np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    b = second/np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    return np.maximum(1-a@b.T, 0)


def patch_descriptors(array):
    x = torch.from_numpy(array.copy()).float()[None]
    patches = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                slices = tuple(slice(n*axis//2, n*(axis+1)//2) for n, axis in zip(array.shape[1:], (i, j, k)))
                patch = x[(slice(None), slice(None), *slices)]
                patches.append(F.adaptive_avg_pool3d(patch, (8, 8, 4)).flatten().numpy())
    return np.stack(patches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='runs/AE-BAKEOFF')
    parser.add_argument('--data', default='data/processed/mrnet-m4-20261021')
    parser.add_argument('--generated', default='runs/M4-MRNET-20261021/generated')
    parser.add_argument('--extractor', default='runs/MEDICALNET-FEATURES/features.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    study = json.loads((Path(args.study)/'study.json').read_text())
    if study['status'] != 'completed':
        raise ValueError('Finish the frozen representation study first')
    snapshot = verify_cohort(args.data)
    if snapshot.fingerprint != study['plan']['dataset_fingerprint']:
        raise ValueError('Feature evaluation cohort does not match the study')
    device = device_for(args.device); torch.set_num_threads(4)
    extractor_path = Path(args.extractor)
    extractor = json.loads((extractor_path.parent/'extractor.json').read_text())
    if sha256_file(extractor_path) != extractor['feature_sha256']:
        raise ValueError('Frozen descriptor export hash mismatch')
    model = torch.jit.load(str(extractor_path), map_location=device).eval()
    out = fresh_output(args.output)
    shape = study['plan']['configs']['ae_kl_aniso']['shape'] if 'ae_kl_aniso' in study['plan']['configs'] else next(iter(study['plan']['configs'].values()))['shape']
    train, valid = (TrainingDataset(args.data, split, shape) for split in ('train', 'validation'))
    train_arrays = [train[i][0].numpy() for i in range(len(train))]
    valid_arrays = [valid[i][0].numpy() for i in range(len(valid))]
    real_features = embeddings(model, train_arrays, device)
    validation_features = embeddings(model, valid_arrays, device)
    records = []
    for record in study['records']:
        folder = Path(args.study)/record['name']/'reconstruction'
        paired = json.loads((folder/'report.json').read_text())
        if (paired['checkpoint_sha256'] != record['checkpoint_sha256'] or paired['dataset_fingerprint'] != snapshot.fingerprint
                or sha256_file(Path(args.study)/record['checkpoint']) != record['checkpoint_sha256']):
            raise ValueError('Paired feature report has stale model/data lineage')
        by_id = {r['scan_id']: r for r in paired['rows']}
        arrays = [generated_volume(folder, by_id[row['scan_id']], shape) for row in valid.rows]
        features = embeddings(model, arrays, device)
        cosine = np.diag(cosine_distances(features, validation_features))
        records.append(dict(arm=record['arm'], seed=record['seed'], mean_paired_feature_cosine_distance=float(cosine.mean()),
                            rows=[dict(scan_id=row['scan_id'], distance=float(distance)) for row, distance in zip(valid.rows, cosine)]))
    generated_root = Path(args.generated)
    manifest = json.loads((generated_root/'manifest.json').read_text())
    if manifest['dataset_fingerprint'] != snapshot.fingerprint:
        raise ValueError('Generation lineage differs from reference cohort')
    generated = [generated_volume(generated_root, r, manifest['shape']) for r in manifest['rows']]
    generated_features = embeddings(model, generated, device)
    np.save(out/'training-features.npy', real_features, allow_pickle=False)
    np.save(out/'validation-features.npy', validation_features, allow_pickle=False)
    np.save(out/'generated-features.npy', generated_features, allow_pickle=False)
    conditions = {tuple(r['conditioning_labels']) for r in manifest['rows'] if r.get('conditioning_labels') is not None}
    matching = [i for i, row in enumerate(train.rows) if row.get('labels') and tuple(json.loads(row['labels'])) in conditions]
    reference_features = real_features[matching] if len(conditions) == 1 and len(matching) >= 2 else real_features
    distribution = feature_distribution_metrics(reference_features, generated_features)
    neighbor_matrix = cosine_distances(generated_features, real_features)
    val_neighbors = cosine_distances(validation_features, real_features).min(1)
    threshold = float(np.quantile(val_neighbors, .01))
    real_patch = np.concatenate([patch_descriptors(x) for x in train_arrays])
    neighbors: list[dict[str, Any]] = []
    for i, volume in enumerate(generated):
        feature_index = int(neighbor_matrix[i].argmin())
        similarities = [reconstruction_metrics(volume[0], real[0])['ssim'] for real in train_arrays]
        ssim_index = int(np.argmax(similarities))
        patch = patch_descriptors(volume)
        distances = np.square(patch[:, None]-real_patch[None]).mean(2)
        patch_index = distances.argmin(1)
        row = dict(file=manifest['rows'][i]['file'], nearest_feature_scan_id=train.rows[feature_index]['scan_id'],
            feature_cosine_distance=float(neighbor_matrix[i, feature_index]),
            suspicious_feature_proximity=bool(neighbor_matrix[i, feature_index] <= threshold),
            nearest_ssim_scan_id=train.rows[ssim_index]['scan_id'], nearest_ssim=float(similarities[ssim_index]),
            patch_neighbors=[dict(query_patch=j, scan_id=train.rows[int(index)//8]['scan_id'],
                                  reference_patch=int(index)%8, pooled_pixel_mse=float(distances[j, index]))
                             for j, index in enumerate(patch_index)],
            exact_file_duplicate=any(manifest['rows'][i]['sha256'] == r.get('output_sha256') for r in train.rows))
        neighbors.append(row)
        slice_grid([volume[0], train_arrays[feature_index][0], train_arrays[ssim_index][0]],
                   ['Generated', 'Nearest frozen medical features', 'Nearest full-volume SSIM'], out/f'neighbors-{i+1}.png')
    report = dict(extractor=extractor, paired_reconstruction=records, generated_neighbors=neighbors,
        smoke_test=any(c.get('smoke_test', False) for c in study['plan']['configs'].values()) or manifest.get('smoke_test', True),
        generated_manifest_sha256=sha256_file(generated_root/'manifest.json'),
        feature_distribution=distribution, distribution_class_matched=len(conditions) == 1 and len(matching) >= 2,
        feature_file_hashes={name: sha256_file(out/f'{name}-features.npy') for name in ('training', 'validation', 'generated')},
        training_reference_count=len(train), generated_count=len(generated),
        real_validation_nearest_feature_distances=val_neighbors.tolist(), proximity_threshold=threshold,
        threshold_protocol='First percentile of same-cohort real validation to training cosine distances, descriptive only',
        patch_protocol='Eight array-grid octants, 8x8x4 pooled intensity descriptors, exhaustive all 96x8 reference patches',
        interpretation='Generic medical features and similarity screens are neither knee-anatomy validation nor a complete privacy/memorization test')
    write_json(out/'feature-memorization-report.json', report)
    print(json.dumps(dict(paired_means=[{k: r[k] for k in ('arm', 'seed', 'mean_paired_feature_cosine_distance')} for r in records],
                          suspicious_generations=sum(r['suspicious_feature_proximity'] for r in neighbors)), indent=2))


if __name__ == '__main__':
    main()
