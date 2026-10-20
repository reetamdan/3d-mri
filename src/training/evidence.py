"""Build submission evidence from actual model artifacts, never hand-entered scores."""
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from src.metrics import downsample_reconstruction_baseline
from src.utils.integrity import sha256_file, generated_volume
from src.utils.logging import write_json
from .common import TrainingDataset, fingerprint, read_checkpoint


def font(size=16):
    try:
        return ImageFont.truetype('arial.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def curve(history, fields, title, path):
    image = Image.new('RGB', (1000, 460), '#0d141c')
    draw = ImageDraw.Draw(image)
    draw.text((30, 12), title, font=font(22), fill='white')
    values = [r[k] for r in history for k in fields if r.get(k) is not None]
    low, high = min(values), max(values)
    high = high if high > low else low + 1
    left, top, width, height = 90, 65, 860, 310
    draw.rectangle((left, top, left+width, top+height), outline='#9aaaba')
    for i in range(5):
        y = top + height*i/4
        draw.line((left, y, left+width, y), fill='#283848')
        draw.text((5, y-8), f'{high-(high-low)*i/4:.4f}', font=font(14), fill='white')
    for i, field in enumerate(fields):
        color = ('#58cbbb', '#f4bd60')[i % 2]
        points = [(left+(r['epoch']-1)*width/max(history[-1]['epoch']-1, 1),
                   top+(high-r[field])*height/(high-low)) for r in history if r.get(field) is not None]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        draw.text((left+i*400, 405), field, font=font(16), fill=color)
    draw.text((left, 380), 'Epoch 1', font=font(14), fill='white')
    draw.text((left+width-95, 380), f'Epoch {history[-1]["epoch"]}', font=font(14), fill='white')
    image.save(path)


def slice_grid(volumes, labels, path):
    """Comparable fixed intensity scale and fixed slice fractions for every row."""
    cell, row_height = 256, 290
    image = Image.new('RGB', (cell*3, row_height*len(volumes)), '#0d141c')
    draw = ImageDraw.Draw(image)
    for row, (volume, label) in enumerate(zip(volumes, labels)):
        draw.text((10, row*row_height+4), label, font=font(17), fill='white')
        for col, fraction in enumerate((.25, .5, .75)):
            index = round((volume.shape[2]-1)*fraction)
            array = np.clip((volume[:, :, index].T+1)*127.5, 0, 255).astype(np.uint8)
            panel = ImageOps.contain(Image.fromarray(array).convert('RGB'), (cell-8, cell-8))
            image.paste(panel, (col*cell+(cell-panel.width)//2, row*row_height+30))
    image.save(path)


def export_evidence(run_dir):
    root = Path(run_dir).resolve()
    status = json.loads((root/'status.json').read_text())
    data = Path(status['data_root'])
    vq_path, diffusion_path = Path(status['vq_checkpoint']), Path(status['diffusion_checkpoint'])
    vq, diffusion = read_checkpoint(vq_path, 'vq'), read_checkpoint(diffusion_path, 'diffusion')
    if vq['smoke_test'] or diffusion['smoke_test']:
        raise ValueError('Smoke runs cannot supply M4 trained-model evidence')
    data_hash = fingerprint(data)
    paired = json.loads((root/'reconstruction/report.json').read_text())
    generated = json.loads((root/'generated/manifest.json').read_text())
    diagnostics = json.loads((root/'generation-report.json').read_text())
    if (paired['dataset_fingerprint'] != data_hash or paired['checkpoint_sha256'] != sha256_file(vq_path)
            or generated['dataset_fingerprint'] != data_hash
            or generated['vq_sha256'] != sha256_file(vq_path)
            or generated['diffusion_sha256'] != sha256_file(diffusion_path)):
        raise ValueError('Evidence lineage does not match the selected models and dataset')
    source_path = Path(status['source_root'])/'source_manifest.json'
    if sha256_file(source_path) != status['source_manifest_sha256']:
        raise ValueError('Source manifest changed after training')
    source = json.loads(source_path.read_text())
    validation = TrainingDataset(data, 'validation', vq['shape'])
    if paired['split'] != 'validation' or not len(validation):
        raise ValueError('M4 evidence requires genuine held-out validation reconstruction')
    baseline = []
    assets = root/'evidence-assets'
    assets.mkdir(exist_ok=True)
    for i, row in enumerate(validation.rows):
        x = validation[i][0][0].numpy()
        interpolated, metrics = downsample_reconstruction_baseline(x, 2.)
        baseline.append(dict(scan_id=row['scan_id'], **metrics))
        match = next(r for r in paired['rows'] if r['scan_id'] == row['scan_id'])
        restored = generated_volume(root/'reconstruction', match, vq['shape'])[0]
        if i < 3:
            slice_grid([x, interpolated, restored], ['Held-out input', 'Half-resolution interpolation', 'Trained VQ reconstruction'],
                       assets/f'validation-{i+1}.png')
    baseline_means = {k: float(np.mean([row[k] for row in baseline])) for k in ('mse', 'mae', 'psnr', 'ssim')}
    write_json(root/'baseline-report.json', dict(split='validation', rows=baseline,
        means=baseline_means, protocol='Half-resolution linear downsample and reconstruct on the same normalized grid'))
    samples = [generated_volume(root/'generated', r, generated['shape'])[0] for r in generated['rows']]
    slice_grid(samples[:4], [f'Generated sample {i+1}; seed {r["seed"]}' for i, r in enumerate(generated['rows'][:4])],
               assets/'generated.png')
    vq_history = json.loads((root/'vq/history.json').read_text())
    diffusion_history = json.loads((root/'diffusion/history.json').read_text())
    curve(vq_history, ['reconstruction_l1', 'validation_l1'], 'VQ reconstruction: training and held-out validation', assets/'vq-training.png')
    curve(diffusion_history, ['noise_mse', 'validation_noise_mse'], 'Latent DDPM: noise-prediction MSE', assets/'diffusion-training.png')
    # Repeat a genuine sample on the same device/model/seed, rather than claiming reproducibility from configuration alone.
    repeat = root/'seed-repeat'
    if not (repeat/'manifest.json').exists():
        from .experiments import generate
        generate(vq_path, diffusion_path, repeat, count=1, seed=generated['rows'][0]['seed'],
                 labels=generated['rows'][0]['conditioning_labels'], device='auto')
    repeated = json.loads((repeat/'manifest.json').read_text())
    generated_volume(repeat, repeated['rows'][0], generated['shape'])
    reproducible = repeated['rows'][0]['sha256'] == generated['rows'][0]['sha256']
    if not reproducible:
        raise ValueError('Same-device repeated sampling did not match the saved sample')
    checkpoint = lambda path, ck, history: dict(file=str(path), sha256=sha256_file(path),
        selected_epoch=ck['epoch'], completed_epochs=history[-1]['epoch'],
        optimizer_steps=sum(r.get('optimizer_steps', 0) for r in history),
        training_seconds=sum(r.get('duration_seconds', 0) for r in history))
    evidence = dict(status='Trained reduced-cohort M4 baseline; independent anatomical validation pending',
        created_at=datetime.now(timezone.utc).isoformat(), smoke_test=False,
        source=dict(url=source['source_url'], upstream_url=source['upstream_url'],
            mirror_license=source['mirror_license'], selection=source['selection'],
            class_counts=source['class_counts'], array_bytes=source['source_array_bytes'],
            manifest_sha256=sha256_file(source_path)),
        data=dict(root=str(data), fingerprint=data_hash, shape=vq['shape'],
            train_count=len(TrainingDataset(data, 'train', vq['shape'])), validation_count=len(validation), test_count=0,
            split_policy=source['split_policy']),
        environment=status['environment'], peak_cuda_allocated_mib=status.get('peak_cuda_allocated_mib'),
        vq=checkpoint(vq_path, vq, vq_history), diffusion=checkpoint(diffusion_path, diffusion, diffusion_history),
        reconstruction=paired['means'], interpolation_baseline=baseline_means,
        generated_count=len(samples), generation=diagnostics,
        sampling_protocol=generated.get('sampler', 'Legacy unbounded DDPM'),
        same_device_seed_repeat_identical=reproducible,
        limitations=['Reduced cohort and resolution; not a full MRNet benchmark',
            'Official validation used for checkpoint selection; no independent hidden-test evaluation',
            'Generated samples remain blurry and do not reliably reproduce coherent large-scale knee anatomy',
            'Generated pathology conditions and anatomical fidelity are not expert verified',
            'No downstream benefit or clinical utility is established by this run'])
    write_json(root/'evidence.json', evidence)
    def number(value):
        return f'{value:.5f}' if isinstance(value, (int, float)) else str(value)
    lines = ['# M3/M4 — Actual experiment evidence', '', f'Generated: {evidence["created_at"]}', '',
        f'**Status:** {evidence["status"]}', '', '## Dataset and hardware', '',
        f'- Source: {source["source_url"]} (version {source["selection"]["version"]}).',
        f'- Source arrays: {source["source_array_bytes"]/1024**2:.1f} MiB; all three acquired planes.',
        f'- Model cohort: {evidence["data"]["train_count"]} train / {len(validation)} validation; sagittal series.',
        f'- Shape: {vq["shape"]}; normalization [-1,1]. Official split assignment preserved.',
        f'- GPU: {status["environment"]["gpu"]}; PyTorch {status["environment"]["torch"]}.',
        f'- Peak allocated CUDA memory: {number(evidence["peak_cuda_allocated_mib"])} MiB.', '',
        '## Trained models', '', '| Stage | Completed epochs | Selected epoch | Optimizer updates | Training seconds |',
        '|---|---:|---:|---:|---:|']
    for name in ('vq', 'diffusion'):
        s = evidence[name]
        lines.append(f'| {name} | {s["completed_epochs"]} | {s["selected_epoch"]} | {s["optimizer_steps"]} | {s["training_seconds"]:.1f} |')
    lines += ['', 'Selection uses internal validation error; last checkpoints remain available for resume.', '',
        '![VQ training](evidence-assets/vq-training.png)', '![Diffusion training](evidence-assets/diffusion-training.png)', '',
        '## Paired held-out reconstruction', '', '| Metric | Trained VQ | Interpolation baseline |', '|---|---:|---:|']
    lines += [f'| {k} | {number(paired["means"][k])} | {number(baseline_means[k])} |' for k in ('mse', 'mae', 'psnr', 'ssim')]
    lines += ['', 'The interpolation baseline retains much more information than random generation; it is a reference, not a trained competitor.', '',
        '![First held-out validation input: fixed slice fractions](evidence-assets/validation-1.png)', '',
        '## Generated-volume diagnostics', '', f'- Generated volumes: {len(samples)}.',
        f'- Sampling protocol: {generated.get("sampler", "Legacy unbounded DDPM")}.',
        f'- Pairwise adaptive 3D MS-SSIM: {number(diagnostics["generated_diversity"]["mean_ms_ssim"])}.',
        f'- Real-training comparison MS-SSIM: {number(diagnostics["real_training_diversity"]["mean_ms_ssim"])}.',
        f'- Protocol: window {diagnostics["generated_diversity"]["window"]}, {diagnostics["generated_diversity"]["levels"]} scales.',
        '- All generated arrays are finite, normalized, shape-checked, and checksum-verified.',
        f'- Same-model, same-device, same-seed repeated sample has identical bytes: {reproducible}.',
        '- PSNR/SSIM are not computed between unrelated generated and real images.', '',
        '![Generated volumes at fixed slice fractions](evidence-assets/generated.png)', '', '## Limitations', '']
    lines += ['- '+s for s in evidence['limitations']]
    lines += ['', '## Artifact verification', '', f'- Dataset fingerprint: `{data_hash}`.',
        f'- Selected VQ SHA-256: `{sha256_file(vq_path)}`.',
        f'- Selected diffusion SHA-256: `{sha256_file(diffusion_path)}`.',
        '- See `evidence.json`, `status.json`, source manifest, and per-stage manifests for complete values.', '']
    (root/'evidence.md').write_text('\n'.join(lines), encoding='utf-8')
    return evidence
