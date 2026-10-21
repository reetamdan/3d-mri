"""Fixed-update, matched-seed MRNet classification augmentation experiment."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F
from .common import TrainingDataset, device_for, fingerprint, fresh_output
from src.utils.seed import set_seed
from src.utils.logging import write_json
from src.utils.integrity import generated_volume
from .protocol import evaluation_context


class MRIClassifier3D(nn.Module):
    def __init__(self, base=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(1, base, 3, 2, 1), nn.GroupNorm(1, base), nn.SiLU(),
            nn.Conv3d(base, base*2, 3, 2, 1), nn.GroupNorm(1, base*2), nn.SiLU(),
            nn.Conv3d(base*2, base*4, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(base*4, 3))

    def forward(self, x):
        return self.net(x)


def classification_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.ndim != 2 or labels.shape[1] != 3 or not len(labels) or labels.shape != probabilities.shape:
        raise ValueError('Expected matching nonempty N by 3 labels and probabilities')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Expected binary labels and finite probabilities in [0,1]')
    result = {}
    for i, name in enumerate(('abnormal', 'acl', 'meniscus')):
        y, p = labels[:, i], probabilities[:, i]
        npos, nneg = int(y.sum()), int(len(y)-y.sum())
        auc = float((rankdata(p)[y == 1].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else None
        result[name] = dict(auc=auc, accuracy=float(((p >= .5) == y).mean()),
                            brier=float(np.mean((p-y)**2)), positive=npos, negative=nneg,
                            threshold=.5,
                            sensitivity=float((p[y == 1] >= .5).mean()) if npos else None,
                            specificity=float((p[y == 0] < .5).mean()) if nneg else None)
    return result


def compare_augmentation(root, generated_dir, output, shape=(128, 128, 32), limit=100,
                         steps=100, seeds=(42, 43, 44), device='auto', smoke_test=False,
                         evaluation_root=None, evaluation_split='validation', bootstrap_samples=500):
    if steps < 1 or limit < 1 or not seeds:
        raise ValueError('Positive training steps, limit, and seeds are required')
    torch.set_num_threads(4)
    device = device_for(device)
    real = TrainingDataset(root, 'train', shape, limit, require_labels=True)
    evaluation_root = evaluation_root or root
    valid = TrainingDataset(evaluation_root, evaluation_split, shape, require_labels=True)
    if not len(real) or not len(valid):
        raise ValueError('Real labeled train and validation splits are required')
    generated_dir = Path(generated_dir)
    manifest = json.loads((generated_dir/'manifest.json').read_text())
    if manifest['dataset_fingerprint'] != fingerprint(root):
        raise ValueError('Generated samples have a different dataset lineage')
    context = evaluation_context(evaluation_root, manifest, evaluation_split, training_root=root)
    if not set(manifest['generator_train_ids']).issubset(real.ids):
        raise ValueError('Generator saw scans outside the limited-data cohort; this would invalidate the augmentation comparison')
    if manifest['smoke_test'] and not smoke_test:
        raise ValueError('Smoke-test generated samples cannot be used in a research run')
    if set(real.ids) & set(valid.ids):
        raise ValueError('Train/validation scan leakage')
    synthetic = []
    for row in manifest['rows']:
        labels = row['conditioning_labels']
        if labels is None or len(labels) != 3 or any(v not in (0, 1) for v in labels):
            raise ValueError('Synthetic augmentation requires conditional generation with recorded binary labels')
        x = torch.from_numpy(generated_volume(generated_dir, row, manifest['shape']).astype(np.float32))
        if x.ndim != 4 or x.shape[0] != 1 or not torch.isfinite(x).all():
            raise ValueError('Invalid synthetic tensor')
        if tuple(x.shape[1:]) != tuple(shape):
            x = F.interpolate(x[None], size=shape, mode='trilinear', align_corners=False)[0]
        synthetic.append((x, torch.tensor(labels, dtype=torch.float32)))
    if not synthetic:
        raise ValueError('No synthetic augmentation samples')
    out = fresh_output(output)
    results = []
    for seed in seeds:
        for arm in ('real_only', 'real_plus_synthetic'):
            set_seed(seed)
            model = MRIClassifier3D().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            rng = np.random.default_rng(seed)
            losses = []
            for step in range(steps):
                # Equal optimizer-update budgets; augmented arm uses 50% synthetic draws.
                if arm == 'real_plus_synthetic' and step % 2:
                    x, y = synthetic[int(rng.integers(len(synthetic)))]
                else:
                    x, y = real[int(rng.integers(len(real)))]
                prediction = model(x[None].to(device))
                loss = F.binary_cross_entropy_with_logits(prediction, y[None].to(device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
            model.eval(); truth, probabilities = [], []
            with torch.no_grad():
                for x, y in valid:
                    truth.append(y.numpy())
                    probabilities.append(model(x[None].to(device)).sigmoid()[0].cpu().numpy())
            truth, probabilities = np.array(truth), np.array(probabilities)
            metrics = classification_metrics(truth, probabilities)
            intervals = classification_intervals(truth, probabilities,
                groups=[r.get('patient_id') or r['scan_id'] for r in valid.rows],
                seed=seed, resamples=bootstrap_samples)
            torch.save(dict(model=model.state_dict(), seed=seed, arm=arm, train_ids=real.ids), out/f'{arm}-{seed}.pt')
            results.append(dict(seed=seed, arm=arm, training_loss=losses, metrics=metrics,
                                confidence_intervals=intervals))
    comparison = {}
    for task in ('abnormal', 'acl', 'meniscus'):
        comparison[task] = {}
        for metric in ('auc', 'accuracy', 'brier'):
            deltas = []
            for seed in seeds:
                arms = {r['arm']: r['metrics'][task][metric] for r in results if r['seed'] == seed}
                if all(v is not None for v in arms.values()):
                    deltas.append(arms['real_plus_synthetic']-arms['real_only'])
            comparison[task][metric] = dict(augmented_minus_real_mean=float(np.mean(deltas)) if deltas else None,
                matched_seed_deltas=deltas, note='Positive favors augmentation for AUC/accuracy; negative for Brier. Descriptive, not a significance test.')
    report = dict(smoke_test=smoke_test, real_train_count=len(real), synthetic_count=len(synthetic),
        confidence_interval_unit='patient' if all(r.get('patient_id') for r in valid.rows) else 'exam; patient identities unavailable',
        comparison=comparison,
        validation_count=len(valid), steps_per_arm=steps, results=results,
        evaluation_split=evaluation_split, evaluation_context=context,
        label_caveat='Synthetic labels are requested conditions, not expert-verified anatomy',
        conclusion='No improvement claim is automated. Inspect matched-seed differences, class support, uncertainty, and artifacts.',
        segmentation='MRNet labels are exam-level. Dice/IoU require a separate source of aligned segmentation masks.')
    write_json(out/'report.json', report)
    return report


def classification_intervals(labels, probabilities, groups=None, seed=42, resamples=500):
    """Patient-cluster bootstrap when IDs exist; otherwise explicitly exam-level."""
    classification_metrics(labels, probabilities)
    if resamples < 100:
        raise ValueError('Use at least 100 bootstrap resamples')
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    identities = np.arange(len(labels)).astype(str) if groups is None else np.asarray(groups).astype(str)
    if identities.shape != (len(labels),):
        raise ValueError('One cluster identity is required per exam')
    clusters = [np.flatnonzero(identities == name) for name in np.unique(identities)]
    values = {task: {metric: [] for metric in ('auc', 'accuracy', 'brier', 'sensitivity', 'specificity')}
              for task in ('abnormal', 'acl', 'meniscus')}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        index = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        metrics = classification_metrics(labels[index], probabilities[index])
        for task in values:
            for metric in values[task]:
                if metrics[task][metric] is not None:
                    values[task][metric].append(metrics[task][metric])
    return {task: {metric: dict(lower=float(np.quantile(draws, .025)) if draws else None,
                               upper=float(np.quantile(draws, .975)) if draws else None,
                               defined_resamples=len(draws), requested_resamples=resamples,
                               clusters=len(clusters), seed=seed, confidence=.95)
                   for metric, draws in metrics.items()} for task, metrics in values.items()}
