"""Exercise VQ-GAN -> latents -> DDPM -> decoded volumes on existing real samples.

Produces explicitly labeled SMOKE_TEST artifacts, not a trained knee model.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.common import model_config
from src.training.experiments import train_vq, cache_latents, train_diffusion, generate, evaluate_reconstruction
from src.training.evaluation import evaluate_generation
from src.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    latest = json.loads((ROOT/'data/metadata/latest_run.json').read_text())
    data = latest['output_dir']
    output = ROOT/'runs'/('SMOKE_TEST-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    c = model_config(ROOT/'configs/smoke-models.yaml')
    vq = train_vq(data, output/'vq', c)
    cache_latents(data, vq, output/'latents', 'cpu')
    diffusion = train_diffusion(output/'latents', output/'diffusion', c)
    generate(vq, diffusion, output/'generated', count=3, device='cpu')
    reconstruction = evaluate_reconstruction(data, vq, output/'reconstruction', split='train', device='cpu')
    generation = evaluate_generation(output/'generated', data, output/'generation-report.json', max_samples=3)
    write_json(output/'summary.json', dict(status='SMOKE TEST ONLY', input_dataset=data,
        reconstruction=reconstruction, generation=generation,
        mrnet_training='Blocked: approved MRNet files unavailable',
        downstream_results='Not run on brain samples: no genuine MRNet labels',
        anatomical_validity='Not established; two epochs only, do not interpret output as realistic MRI'))
    write_json(ROOT/'runs/latest_smoke.json', {'output_dir': str(output)})
    print(f'Smoke artifacts: {output}')


if __name__ == '__main__':
    main()
