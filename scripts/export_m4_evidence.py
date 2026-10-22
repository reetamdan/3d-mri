"""Rebuild measured M3/M4 figures and evidence without retraining."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.evidence import export_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/M4-MRNET-20261021')
    args = parser.parse_args()
    result = export_evidence(args.run)
    print(json.dumps({k: result[k] for k in ('status', 'reconstruction', 'same_device_seed_repeat_identical')}, indent=2))


if __name__ == '__main__':
    main()
