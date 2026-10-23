"""Fair conventional/resampling/synthetic comparisons; synthetic runs require a quality gate."""
import argparse
import json
from pathlib import Path
import sys
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.downstream_fair import compare_downstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--generated')
    parser.add_argument('--generation-gate')
    parser.add_argument('--development-data')
    parser.add_argument('--evaluation-data')
    parser.add_argument('--evaluation-split', choices=['validation', 'test'], default='test')
    parser.add_argument('--arms', nargs='+', default=['real_only', 'torchio', 'resample'])
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    result = compare_downstream(args.data, args.output, arms=tuple(args.arms), generated=args.generated,
        generation_gate=args.generation_gate, development_root=args.development_data,
        evaluation_root=args.evaluation_data, evaluation_split=args.evaluation_split,
        steps=args.steps, batch_size=args.batch_size, seeds=tuple(args.seeds), device=args.device)
    print(json.dumps(dict(aggregate=result['aggregate'], evaluation_role=result['evaluation_role']), indent=2))


if __name__ == '__main__':
    main()
