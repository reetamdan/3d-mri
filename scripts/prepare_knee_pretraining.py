"""Prepare already-authorized fastMRI/SKM-TEA image targets for representation pretraining."""
import argparse
import json
from pathlib import Path
import sys
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.knee_pretraining import prepare_knee_pretraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', choices=['fastmri', 'skm-tea'], required=True)
    parser.add_argument('--acquisition', required=True)
    parser.add_argument('--image-key', required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=[128, 128, 32])
    args = parser.parse_args()
    result = prepare_knee_pretraining(args.source, args.manifest, args.output, dataset=args.dataset,
                                      acquisition=args.acquisition, image_key=args.image_key, shape=args.shape)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
