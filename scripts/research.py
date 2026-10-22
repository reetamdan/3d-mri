"""MRI data, model training, sampling and auditable research evaluation commands."""
import argparse
import json
import sys
from pathlib import Path
if __name__ == '__main__' and not __package__:
    # Compatibility for direct script invocation; the installed CLI uses packages.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.common import model_config
from mri3d.configuration import read_configuration
from mri3d.presets import PRESETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    config = sub.add_parser('show-config', help='Print an installed configuration preset as JSON')
    config.add_argument('--preset', choices=PRESETS, default='models.yaml')
    mr = sub.add_parser('prepare-mrnet')
    mr.add_argument('--root', required=True); mr.add_argument('--output', required=True)
    mr.add_argument('--plane', choices=['axial', 'sagittal', 'coronal'], default='sagittal')
    mr.add_argument('--shape', nargs=3, type=int, default=[128, 128, 32]); mr.add_argument('--limit', type=int)
    for stage in ('train-vq', 'train-diffusion'):
        p = sub.add_parser(stage)
        p.add_argument('--data', required=True); p.add_argument('--output', required=True)
        p.add_argument('--config', help='YAML file override; defaults to the bundled models.yaml preset')
        p.add_argument('--resume', action='store_true')
    p = sub.add_parser('cache-latents')
    p.add_argument('--data', required=True); p.add_argument('--vq', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p = sub.add_parser('generate')
    p.add_argument('--vq', required=True); p.add_argument('--diffusion', required=True); p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=4); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--labels', nargs=3, type=int); p.add_argument('--device', default='auto')
    p.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm')
    p.add_argument('--sampling-steps', type=int); p.add_argument('--guidance-scale', type=float, default=1.)
    p.add_argument('--eta', type=float, default=0.); p.add_argument('--weights', choices=['auto', 'raw', 'ema'], default='auto')
    p = sub.add_parser('evaluate-reconstruction')
    p.add_argument('--data', required=True); p.add_argument('--vq', required=True); p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation'); p.add_argument('--device', default='auto')
    p.add_argument('--training-data'); p.add_argument('--protocol')
    p = sub.add_parser('evaluate-generation')
    p.add_argument('--data', required=True); p.add_argument('--generated', required=True); p.add_argument('--output', required=True)
    p.add_argument('--max-samples', type=int, default=16)
    p.add_argument('--training-data'); p.add_argument('--real-split', choices=['train', 'validation', 'test'], default='train')
    p = sub.add_parser('freeze-protocol')
    p.add_argument('--training-data', required=True); p.add_argument('--evaluation-data', required=True)
    p.add_argument('--output', required=True); p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--primary-metric', default='ssim'); p.add_argument('--hypothesis', required=True)
    p = sub.add_parser('tokenizer-diagnostics')
    p.add_argument('--data', required=True); p.add_argument('--vq', required=True); p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='train'); p.add_argument('--device', default='auto')
    p = sub.add_parser('evaluate-features')
    p.add_argument('--real', required=True); p.add_argument('--generated', required=True)
    p.add_argument('--extractor-id', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('relocate')
    p.add_argument('--root', default='.')
    p = sub.add_parser('export-inference')
    p.add_argument('--vq', required=True); p.add_argument('--diffusion', required=True); p.add_argument('--output', required=True)
    p.add_argument('--weights', choices=['auto', 'raw', 'ema'], default='auto')
    p = sub.add_parser('prepare-review')
    p.add_argument('--data', required=True); p.add_argument('--generated', required=True); p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=8); p.add_argument('--seed', type=int, default=42)
    p = sub.add_parser('summarize-review')
    p.add_argument('--packet', required=True); p.add_argument('--scores', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('compare-augmentation')
    p.add_argument('--data', required=True); p.add_argument('--generated', required=True); p.add_argument('--output', required=True)
    p.add_argument('--shape', nargs=3, type=int, default=[128, 128, 32]); p.add_argument('--limit', type=int, default=100)
    p.add_argument('--steps', type=int, default=100); p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--device', default='auto'); p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--evaluation-data'); p.add_argument('--evaluation-split', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    from src.training import experiments as e
    if args.command == 'show-config':
        result = read_configuration(preset=args.preset)
    elif args.command == 'prepare-mrnet':
        from src.data.mrnet import prepare_mrnet
        result = prepare_mrnet(args.root, args.output, args.plane, args.shape, args.limit)
    elif args.command == 'train-vq':
        result = e.train_vq(args.data, args.output, model_config(args.config), args.resume)
    elif args.command == 'train-diffusion':
        result = e.train_diffusion(args.data, args.output, model_config(args.config), args.resume)
    elif args.command == 'cache-latents':
        result = e.cache_latents(args.data, args.vq, args.output, args.device)
    elif args.command == 'generate':
        result = e.generate(args.vq, args.diffusion, args.output, args.count, args.seed, args.labels, args.device,
                            sampler=args.sampler, sampling_steps=args.sampling_steps, guidance_scale=args.guidance_scale,
                            eta=args.eta, weights=args.weights)
    elif args.command == 'evaluate-reconstruction':
        result = e.evaluate_reconstruction(args.data, args.vq, args.output, args.split, args.device,
                                            training_root=args.training_data, protocol=args.protocol)
    elif args.command == 'evaluate-generation':
        from src.training.evaluation import evaluate_generation
        result = evaluate_generation(args.generated, args.data, args.output, args.max_samples,
                                     training_root=args.training_data, real_split=args.real_split)
    elif args.command == 'freeze-protocol':
        from src.training.protocol import freeze_protocol
        result = freeze_protocol(args.training_data, args.evaluation_data, args.output, args.split,
                                 args.seeds, args.primary_metric, args.hypothesis)
    elif args.command == 'tokenizer-diagnostics':
        from src.training.common import TrainingDataset, device_for
        from src.training.diagnostics import tokenizer_diagnostics
        from src.utils.logging import write_json
        device = device_for(args.device)
        model, ck = e.load_vq(args.vq, device)
        result = tokenizer_diagnostics(model, TrainingDataset(args.data, args.split, ck['shape']), device)
        write_json(args.output, result)
    elif args.command == 'evaluate-features':
        import numpy as np
        from src.training.statistics import feature_distribution_metrics
        from src.utils.integrity import sha256_file
        from src.utils.logging import write_json
        result = feature_distribution_metrics(np.load(args.real, allow_pickle=False), np.load(args.generated, allow_pickle=False))
        result.update(extractor_id=args.extractor_id, real_features_sha256=sha256_file(args.real),
                      generated_features_sha256=sha256_file(args.generated),
                      protocol='Caller-supplied exam-level embeddings; extractor suitability must be established separately')
        write_json(args.output, result)
    elif args.command == 'relocate':
        from src.utils.artifacts import relocate_project
        result = relocate_project(args.root)
    elif args.command == 'export-inference':
        from src.training.export import export_inference_pair
        result = export_inference_pair(args.vq, args.diffusion, args.output, args.weights)
    elif args.command == 'prepare-review':
        from src.training.review import prepare_review
        result = prepare_review(args.data, args.generated, args.output, args.count, args.seed)
    elif args.command == 'summarize-review':
        from src.training.review import summarize_review
        result = summarize_review(args.packet, args.scores, args.output)
    else:
        from src.training.downstream import compare_augmentation
        result = compare_augmentation(args.data, args.generated, args.output, args.shape,
            args.limit, args.steps, args.seeds, args.device, args.smoke_test,
            evaluation_root=args.evaluation_data, evaluation_split=args.evaluation_split)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
