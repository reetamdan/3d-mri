"""Matched-seed EMA/CFG/sampler diagnostics and conditioning changes; not expert review."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.experiments import generate
from src.training.evaluation import evaluate_generation
from src.training.review import prepare_review
from src.utils.logging import write_json
from src.utils.integrity import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='runs/RESEARCH-V2/upgraded-42')
    p.add_argument('--data', default='data/processed/research-v2')
    p.add_argument('--output', default='runs/RESEARCH-V2/sampling-ablation')
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.count < 2 or args.threads < 1:
        p.error('Use at least two samples and positive threads')
    torch.set_num_threads(args.threads)
    run, output = Path(args.run), Path(args.output)
    vq_info = json.loads((run/'vq/run.json').read_text())
    diffusion_info = json.loads((run/'diffusion/run.json').read_text())
    vq, diffusion = run/'vq'/vq_info['checkpoint_file'], run/'diffusion'/diffusion_info['checkpoint_file']
    arms = {
        'raw-ddpm': dict(weights='raw'),
        'ema-ddpm': dict(weights='ema'),
        'ema-ddim-50': dict(weights='ema', sampler='ddim', sampling_steps=50),
        'ema-ddim-50-cfg': dict(weights='ema', sampler='ddim', sampling_steps=50, guidance_scale=1.5),
    }
    results = []
    for name, options in arms.items():
        folder = output/name
        before = time.perf_counter()
        reused = (folder/'manifest.json').exists()
        if reused:
            previous = json.loads((folder/'manifest.json').read_text())
            if (len(previous['rows']) != args.count or previous['vq_sha256'] != sha256_file(vq)
                    or previous['diffusion_sha256'] != sha256_file(diffusion)):
                raise ValueError('Existing ablation has different models/count; select a new output folder')
            settings = previous.get('sampling', {})
            if (settings.get('method', 'ddpm') != options.get('sampler', 'ddpm')
                    or settings.get('weights') != options['weights']
                    or settings.get('guidance_scale', 1.) != options.get('guidance_scale', 1.)
                    or any(r['seed'] != 2026+i or r['conditioning_labels'] != [1, 0, 0]
                           for i, r in enumerate(previous['rows']))):
                raise ValueError('Existing ablation protocol differs from the declared arm')
        else:
            generate(vq, diffusion, folder, count=args.count, seed=2026, labels=[1, 0, 0], device='cuda', **options)
        report = evaluate_generation(folder, args.data, output/f'{name}.json', max_samples=args.count, real_split='validation')
        results.append(dict(name=name, options=options, elapsed_this_invocation_seconds=time.perf_counter()-before,
                            generation_reused=reused,
                            diagnostics=report))
    alternate = output/'acl-request'
    if not (alternate/'manifest.json').exists():
        generate(vq, diffusion, alternate, count=args.count, seed=2026, labels=[1, 1, 0], device='cuda',
                 weights='ema', sampler='ddim', sampling_steps=50, guidance_scale=1.5)
    if not (output/'blinded-review').exists():
        prepare_review(args.data, output/'ema-ddim-50-cfg', output/'blinded-review', count=args.count)
    write_json(output/'ablation.json', dict(arms=results, fixed_seeds=[2026+i for i in range(args.count)],
        alternative_condition=[1, 1, 0], original_condition=[1, 0, 0],
        label_fidelity_verified=False, expert_review_completed=False,
        interpretation='Inference ablation with fixed checkpoints and initial seeds. Texture/nearest-neighbor metrics do not establish anatomy or pathology.'))
    print('Sampling ablation and blank blinded-review packet complete', flush=True)


if __name__ == '__main__':
    main()
