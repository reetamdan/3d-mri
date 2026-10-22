"""Train and evaluate the real MRNet M4 baseline; stages are explicit and resumable."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.data.mrnet import prepare_mrnet
from src.training.common import model_config, fingerprint
from src.training.experiments import train_vq, cache_latents, train_diffusion, generate, evaluate_reconstruction
from src.training.evaluation import evaluate_generation
from src.utils.integrity import sha256_file, confined_path
from src.utils.logging import write_json
from src.utils.artifacts import reference, resolve_reference


def run(args):
    source, data, output = map(Path, (args.source, args.data, args.output))
    vq_config = model_config(args.vq_config, preset='m4-vq.yaml')
    diffusion_config = model_config(args.diffusion_config, preset='m4-diffusion.yaml')
    if vq_config['smoke_test'] or diffusion_config['smoke_test']:
        raise ValueError('The M4 runner requires real training configurations')
    if vq_config['shape'] != diffusion_config['shape']:
        raise ValueError('VQ and diffusion shapes must match')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Experiment exists; choose a new output or use --resume')
    manifest_path = source/'source_manifest.json'
    source_manifest = json.loads(manifest_path.read_text())
    if not source_manifest['complete']:
        raise ValueError('Finish the source subset download first')
    for row in source_manifest['files']:
        if sha256_file(confined_path(source, row['file'])) != row['sha256']:
            raise ValueError('Source checksum mismatch; download the subset again')
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status = dict(started_at=datetime.now(timezone.utc).isoformat(), stages={},
        source_manifest_sha256=sha256_file(manifest_path), source_root=reference(source, output),
        data_root=reference(data, output), configs=dict(vq=vq_config, diffusion=diffusion_config),
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    status_path = output/'status.json'
    previous_peak_mib = 0
    if args.resume and status_path.exists():
        old = json.loads(status_path.read_text())
        if old['source_manifest_sha256'] != status['source_manifest_sha256'] or old['configs'] != status['configs']:
            raise ValueError('Resume requires the same source manifest and stage configurations')
        status['previous_sessions'] = old.get('previous_sessions', []) + [
            {k: old[k] for k in ('started_at', 'finished_at', 'session_seconds', 'stages', 'refinement') if k in old}]
        previous_peak_mib = old.get('peak_cuda_allocated_mib') or 0
    def stage(name, action):
        status['current_stage'] = name
        write_json(status_path, status)
        print(f'\n=== {name} ===', flush=True)
        before = time.perf_counter()
        result = action()
        status['stages'][name] = dict(seconds=time.perf_counter()-before, completed=True)
        write_json(status_path, status)
        return result
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if not data.exists():
        stage('preprocess', lambda: prepare_mrnet(source, data, 'sagittal', vq_config['shape']))
    else:
        experiment = json.loads((data/'experiment.json').read_text())
        if resolve_reference(data, experiment['root']) != source.resolve() or experiment['shape'] != vq_config['shape']:
            raise ValueError('Existing processed dataset has different source or geometry')
        fingerprint(data)
    vq = stage('train_vq', lambda: train_vq(data, output/'vq', vq_config,
                resume=args.resume and (output/'vq/last.pt').exists()))
    if not (output/'latents/manifest.json').exists():
        stage('cache_latents', lambda: cache_latents(data, vq, output/'latents', vq_config['device']))
    else:
        cached = json.loads((output/'latents/manifest.json').read_text())
        if cached['vq_sha256'] != sha256_file(vq) or cached['dataset_fingerprint'] != fingerprint(data):
            raise ValueError('Existing cache no longer matches the selected VQ model; choose a new experiment')
    diffusion = stage('train_diffusion', lambda: train_diffusion(output/'latents', output/'diffusion',
                diffusion_config, resume=args.resume and (output/'diffusion/last.pt').exists()))
    if not (output/'reconstruction/report.json').exists():
        stage('held_out_reconstruction', lambda: evaluate_reconstruction(data, vq,
            output/'reconstruction', device=vq_config['device']))
    if not (output/'generated/manifest.json').exists():
        stage('generation', lambda: generate(vq, diffusion, output/'generated', count=args.count,
              labels=[1, 0, 0] if diffusion_config['conditional'] else None,
              seed=2026, device=diffusion_config['device']))
    stage('generation_diagnostics', lambda: evaluate_generation(output/'generated', data,
          output/'generation-report.json', max_samples=args.count))
    status.update(finished_at=datetime.now(timezone.utc).isoformat(), session_seconds=time.perf_counter()-start,
        peak_cuda_allocated_mib=max(previous_peak_mib, torch.cuda.max_memory_allocated()/1024**2) if torch.cuda.is_available() else None,
        current_stage='complete', vq_checkpoint=reference(vq, output),
        diffusion_checkpoint=reference(diffusion, output), dataset_fingerprint=fingerprint(data))
    write_json(status_path, status)
    # Evidence export is separate so figures/documents can be regenerated without retraining.
    from src.training.evidence import export_evidence
    stage('evidence', lambda: export_evidence(output))
    status['current_stage'] = 'complete'
    write_json(status_path, status)
    write_json(output.parent/'latest_m4.json', dict(output_dir=reference(output, output.parent)))
    print(f'\nCompleted trained-model baseline: {output.resolve()}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='data/mrnet/MRNet-v1.0')
    p.add_argument('--data', default='data/processed/mrnet-m4-20261021')
    p.add_argument('--output', default='runs/M4-MRNET-20261021')
    p.add_argument('--vq-config', help='VQ YAML override; defaults to the bundled preset')
    p.add_argument('--diffusion-config', help='Diffusion YAML override; defaults to the bundled preset')
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.count < 2:
        p.error('Generate at least two samples for diversity diagnostics')
    run(args)


if __name__ == '__main__':
    main()
