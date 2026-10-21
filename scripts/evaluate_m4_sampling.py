"""Retain the initial unbounded ablation and evaluate the bounded DDPM sampler."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.experiments import generate, train_diffusion
from src.training.evaluation import evaluate_generation
from src.training.evidence import export_evidence
from src.utils.logging import write_json
from src.utils.artifacts import resolve_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='runs/M4-MRNET-20261021')
    p.add_argument('--refine-epochs', type=int, help='Resume the same diffusion model to this total epoch count')
    args = p.parse_args()
    root = Path(args.run)
    status = json.loads((root/'status.json').read_text())
    for key in ('data_root', 'vq_checkpoint', 'diffusion_checkpoint'):
        status[key] = str(resolve_reference(root, status[key]))
    if args.refine_epochs is not None and args.refine_epochs <= status['configs']['diffusion']['epochs']:
        raise ValueError('Refinement must increase total training epochs; no artifacts were moved')
    old_manifest = root/'generated/manifest.json'
    if old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        if old.get('sampler') and args.refine_epochs is None:
            raise ValueError('Bounded sampling is already evaluated; choose a new experiment for a new protocol')
        name = f'bounded-{status["configs"]["diffusion"]["epochs"]}-epochs' if old.get('sampler') else 'unbounded'
        archive = root/'sampling-ablation'/name
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(status['diffusion_checkpoint'], archive/'diffusion-checkpoint.pt')
        shutil.copy2(root/'diffusion/history.json', archive/'diffusion-history.json')
        for name in ('generated', 'seed-repeat', 'generation-report.json', 'evidence.json', 'evidence.md', 'evidence-assets'):
            if (root/name).exists():
                shutil.move(str(root/name), str(archive/name))
        write_json(archive/'note.json', dict(reason='Sampling/training refinement; previous actual outputs retained for comparison',
            protocol=old.get('sampler', 'Unbounded predicted-clean latent'), diffusion_sha256=old['diffusion_sha256']))
    if args.refine_epochs is not None:
        c = dict(status['configs']['diffusion'])
        before = time.perf_counter()
        previous_epochs = c['epochs']
        c['epochs'] = args.refine_epochs
        checkpoint = train_diffusion(root/'latents', root/'diffusion', c, resume=True)
        status['configs']['diffusion'] = c
        status['diffusion_checkpoint'] = str(Path(checkpoint).resolve())
        status['refinement'] = dict(previous_epochs=previous_epochs, total_epochs=c['epochs'],
                                    seconds=time.perf_counter()-before,
                                    reason='Initial samples lacked coherent anatomy despite low denoising loss')
        write_json(root/'status.json', status)
    generate(status['vq_checkpoint'], status['diffusion_checkpoint'], root/'generated', count=8,
             seed=2026, labels=[1, 0, 0], device='cuda')
    evaluate_generation(root/'generated', status['data_root'], root/'generation-report.json', max_samples=8)
    export_evidence(root)
    status['current_stage'] = 'complete'
    status['sampling_protocol'] = 'Training-codebook-bounded predicted-clean DDPM posterior; original unbounded ablation retained'
    write_json(root/'status.json', status)


if __name__ == '__main__':
    main()
