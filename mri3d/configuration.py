"""Shared file/preset loading for source checkouts and installed distributions."""
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from mri3d.presets import PRESETS


def read_configuration(path: str | Path | None = None, *, preset: str = 'models.yaml') -> dict[str, Any]:
    if path:
        text = Path(path).read_text(encoding='utf-8')
    else:
        if preset not in PRESETS:
            raise ValueError(f'Unknown bundled preset: {preset}')
        resource = files('mri3d.presets').joinpath(preset)
        # Editable/source installs use the same canonical files copied by BuildPy.
        text = resource.read_text(encoding='utf-8') if resource.is_file() else (
            Path(__file__).resolve().parents[1]/'configs'/preset).read_text(encoding='utf-8')
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a YAML mapping')
    return value


def workspace_root() -> Path:
    checkout = Path(__file__).resolve().parents[1]
    return checkout if (checkout/'configs/config.yaml').is_file() else Path.cwd().resolve()
