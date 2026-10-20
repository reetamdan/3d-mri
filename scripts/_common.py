import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config


def configuration(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/config.yaml'))
    return load_config(parser.parse_args().config)
