from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from src.training.common import model_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('changes', [
    {'epochs': 1.5}, {'batch_size': True}, {'shape': [16, 16, 7]},
    {'learning_rate': float('nan')}, {'learning_rate': float('inf')},
    {'seed': -1}, {'prediction_type': 'unknown'}, {'gradient_accumulation': 0},
    {'label_dropout': 1.}, {'ema_decay': 1.}, {'diffusion_steps': '200'},
    {'validation_draws': 10000}, {'channel_mult': [1, -2]},
    {'mixed_precision': 'false'}, {'gan_weight': float('nan')},
])
def test_model_config_rejects_invalid_values(tmp_path, changes):
    original = yaml.safe_load((ROOT/'configs/research-v2.yaml').read_text())
    config = deepcopy(original); config.update(changes)
    path = tmp_path/'invalid.yaml'; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        model_config(path)


def test_model_config_rejects_non_mapping_and_invalid_codebook(tmp_path):
    path = tmp_path/'invalid.yaml'; path.write_text('[]')
    with pytest.raises(ValueError, match='mapping'):
        model_config(path)
    config = yaml.safe_load((ROOT/'configs/research-v2.yaml').read_text())
    config['vq']['codes'] = 1
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='at least two'):
        model_config(path)
