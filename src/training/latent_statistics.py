"""Generic frozen VQ/KL caches with train-only per-channel normalization."""
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.autoencoder import encode_latent, KLAutoencoder3D
from src.models.vqvae import VQVAE3D
from src.utils.artifacts import reference
from src.utils.cohort import verify_cohort
from src.utils.integrity import sha256_file
from src.utils.logging import write_json
from .common import TrainingDataset, fresh_output, device_for
from .representation import load_representation
from .latents import pack_latents, transform_latents as transform_latents


@torch.no_grad()
def cache_representation(root, checkpoint, output, device='auto', representation_gate=None) -> dict[str, Any]:
    device = device_for(device)
    model, ck = load_representation(checkpoint, device)
    cohort = verify_cohort(root)
    if cohort.fingerprint != ck['dataset_fingerprint']:
        raise ValueError('Autoencoder cache cohort differs from checkpoint lineage')
    gate = None
    if representation_gate:
        gate = json.loads(Path(representation_gate).read_text())
        if (gate.get('autoencoder_sha256') != sha256_file(checkpoint) or gate.get('quantitative_gate_pass') is not True
                or gate.get('visual_review_pass') is not True or gate.get('dataset_fingerprint') != cohort.fingerprint):
            raise ValueError('Representation gate is failed, incomplete, or belongs to different weights')
    out = fresh_output(output)
    rows = []
    channels = ck['model_config']['latent_channels']
    sums, squares = torch.zeros(channels, dtype=torch.float64), torch.zeros(channels, dtype=torch.float64)
    lower, upper = torch.full((channels,), float('inf')), torch.full((channels,), -float('inf'))
    count = 0
    by_condition: dict[str, list[list[float]]] = defaultdict(list)
    sigma_means = []
    code_counts = torch.zeros(model.quantizer.embedding.num_embeddings, dtype=torch.float64) if isinstance(model, VQVAE3D) else None
    class_counts: dict[str, torch.Tensor] = {}
    for split in ('train', 'validation', 'test'):
        dataset = TrainingDataset(root, split, ck['shape'], strict_shape=True)
        (out/split).mkdir()
        for i, row in enumerate(dataset.rows):
            if split == 'train' and row['scan_id'] not in ck['train_ids']:
                continue
            x, labels = dataset[i]
            batch = x[None].to(device)
            if isinstance(model, KLAutoencoder3D):
                z, sigma = model.encode(batch)
                if split == 'train':
                    sigma_means.append(float(sigma.mean()))
            else:
                z = encode_latent(model, batch)
            tensor = z[0].float().cpu()
            relative = f'{split}/{row["scan_id"]}.pt'
            torch.save(tensor, out/relative)
            rows.append(dict(scan_id=row['scan_id'], split=split, labels=labels.tolist(), file=relative,
                             sha256=sha256_file(out/relative)))
            if split == 'train':
                values = tensor.double().flatten(1)
                sums += values.sum(1); squares += values.square().sum(1)
                lower, upper = torch.minimum(lower, values.amin(1)), torch.maximum(upper, values.amax(1))
                count += values.shape[1]
                condition = ','.join(str(int(y)) for y in labels)
                by_condition[condition].append(values.mean(1).tolist())
                if isinstance(model, VQVAE3D) and code_counts is not None:
                    counts = torch.bincount(model.quantizer.last_indices.cpu(), minlength=len(code_counts)).double()
                    code_counts += counts
                    class_counts.setdefault(condition, torch.zeros_like(counts)).add_(counts)
    if count == 0:
        raise ValueError('No training latents to fit scaling')
    mean = sums/count
    std = (squares/count-mean.square()).clamp_min(1e-12).sqrt()
    codebook = None
    if code_counts is not None:
        probabilities = code_counts/code_counts.sum()
        entropy = float(-(probabilities*(probabilities+1e-15).log()).sum())
        codebook = dict(counts=code_counts.tolist(), used_codes=int((code_counts > 0).sum()),
                        dead_fraction=float((code_counts == 0).double().mean()), entropy_nats=entropy,
                        perplexity=float(np.exp(entropy)), by_exam_label={key: value.tolist() for key, value in class_counts.items()})
    manifest = dict(rows=rows, packed=pack_latents(out, rows), latent_shape=list(tensor.shape), shape=ck['shape'],
        codebook_diagnostics=codebook,
        mean=float(sums.sum()/(count*len(sums))),
        std=float((squares.sum()/(count*len(sums))-(sums.sum()/(count*len(sums)))**2).clamp_min(1e-12).sqrt()),
        channel_mean=mean.tolist(), channel_std=std.tolist(), channel_min=lower.tolist(), channel_max=upper.tolist(),
        normalization_fit='train_only_per_channel', posterior_cache='mean' if isinstance(model, KLAutoencoder3D) else 'quantized',
        posterior_sigma_mean=float(np.mean(sigma_means)) if sigma_means else None,
        condition_pooled_channel_statistics={k: dict(exams=len(v), mean=np.mean(v, axis=0).tolist(), std=np.std(v, axis=0).tolist())
                                             for k, v in by_condition.items()},
        vq_sha256=sha256_file(checkpoint), autoencoder_sha256=sha256_file(checkpoint),
        vq_path=reference(checkpoint, out), autoencoder_family='kl' if isinstance(model, KLAutoencoder3D) else 'vq',
        dataset_fingerprint=cohort.fingerprint, train_ids=ck['train_ids'], cohort_records=cohort.records(),
        representation_pretraining_records=ck.get('representation_pretraining_records', []),
        representation_gate=gate,
        representation_gate_sha256=sha256_file(representation_gate) if representation_gate else None,
        smoke_test=ck['smoke_test'])
    write_json(out/'manifest.json', manifest)
    return manifest
