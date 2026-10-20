"""CLI for milestones 1–5. No action runs automatically on import."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.common import model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    mr = sub.add_parser('prepare-mrnet')
    mr.add_argument('--root', required=True); mr.add_argument('--output', required=True)
    mr.add_argument('--plane', choices=['axial', 'sagittal', 'coronal'], default='sagittal')
    mr.add_argument('--shape', nargs=3, type=int, default=[128, 128, 32]); mr.add_argument('--limit', type=int)
    for stage in ('train-vq', 'train-diffusion'):
        p = sub.add_parser(stage)
        p.add_argument('--data', required=True); p.add_argument('--output', required=True)
        p.add_argument('--config', default='configs/models.yaml'); p.add_argument('--resume', action='store_true')
    p = sub.add_parser('cache-latents')
    p.add_argument('--data', required=True); p.add_argument('--vq', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p = sub.add_parser('generate')
    p.add_argument('--vq', required=True); p.add_argument('--diffusion', required=True); p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=4); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--labels', nargs=3, type=int); p.add_argument('--device', default='auto')
    p = sub.add_parser('evaluate-reconstruction')
    p.add_argument('--data', required=True); p.add_argument('--vq', required=True); p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation'); p.add_argument('--device', default='auto')
    p = sub.add_parser('evaluate-generation')
    p.add_argument('--data', required=True); p.add_argument('--generated', required=True); p.add_argument('--output', required=True)
    p.add_argument('--max-samples', type=int, default=16)
    p = sub.add_parser('compare-augmentation')
    p.add_argument('--data', required=True); p.add_argument('--generated', required=True); p.add_argument('--output', required=True)
    p.add_argument('--shape', nargs=3, type=int, default=[128, 128, 32]); p.add_argument('--limit', type=int, default=100)
    p.add_argument('--steps', type=int, default=100); p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--device', default='auto'); p.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    from src.training import experiments as e
    if args.command == 'prepare-mrnet':
        from src.data.mrnet import prepare_mrnet
        result = prepare_mrnet(args.root, args.output, args.plane, args.shape, args.limit)
    elif args.command == 'train-vq':
        result = e.train_vq(args.data, args.output, model_config(args.config), args.resume)
    elif args.command == 'train-diffusion':
        result = e.train_diffusion(args.data, args.output, model_config(args.config), args.resume)
    elif args.command == 'cache-latents':
        result = e.cache_latents(args.data, args.vq, args.output, args.device)
    elif args.command == 'generate':
        result = e.generate(args.vq, args.diffusion, args.output, args.count, args.seed, args.labels, args.device)
    elif args.command == 'evaluate-reconstruction':
        result = e.evaluate_reconstruction(args.data, args.vq, args.output, args.split, args.device)
    elif args.command == 'evaluate-generation':
        from src.training.evaluation import evaluate_generation
        result = evaluate_generation(args.generated, args.data, args.output, args.max_samples)
    else:
        from src.training.downstream import compare_augmentation
        result = compare_augmentation(args.data, args.generated, args.output, args.shape,
            args.limit, args.steps, args.seeds, args.device, args.smoke_test)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
