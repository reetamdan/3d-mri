"""End-to-end evidence validation using explicitly artificial numerical fixtures."""
import json
from pathlib import Path
import torch
from src.training.common import model_config
from src.training.experiments import train_vq, cache_latents, train_diffusion, generate, evaluate_reconstruction
from src.training.evaluation import evaluate_generation
from src.training.evidence import export_evidence
from src.utils.artifacts import reference
from src.utils.integrity import sha256_file
from src.utils.logging import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_evidence_export_reads_relative_artifact_references(labeled_tensors, tmp_path):
    source = tmp_path/'test-only-source'; source.mkdir()
    source_manifest = source/'source_manifest.json'
    write_json(source_manifest, dict(source_url='fixture://artificial-test-only', upstream_url='fixture://none',
        mirror_license='Not real MRI data', selection={'version': 1}, class_counts={}, source_array_bytes=0,
        split_policy='Artificial disjoint software-test identities'))
    config = model_config(ROOT/'configs/smoke-models.yaml')
    config.update(shape=[16]*3, epochs=1, steps_per_epoch=1, smoke_test=False, conditional=True,
                  select_best=True, gan_weight=0, diffusion_steps=4)
    run = tmp_path/'experiment'
    vq = train_vq(labeled_tensors, run/'vq', config)
    cache_latents(labeled_tensors, vq, run/'latents', 'cpu')
    diffusion = train_diffusion(run/'latents', run/'diffusion', config)
    evaluate_reconstruction(labeled_tensors, vq, run/'reconstruction', device='cpu')
    generate(vq, diffusion, run/'generated', count=2, labels=[1, 1, 1], device='cpu', sampler='ddim', sampling_steps=2)
    evaluate_generation(run/'generated', labeled_tensors, run/'generation-report.json', max_samples=2)
    write_json(run/'status.json', dict(data_root=reference(labeled_tensors, run), source_root=reference(source, run),
        source_manifest_sha256=sha256_file(source_manifest), vq_checkpoint=reference(vq, run),
        diffusion_checkpoint=reference(diffusion, run), peak_cuda_allocated_mib=None,
        environment={'gpu': 'CPU numerical test fixture', 'torch': str(torch.__version__)}))
    report = export_evidence(run)
    assert report['same_device_seed_repeat_identical']
    assert report['data']['validation_count'] == 2
    assert (run/'evidence-assets/validation-1.png').is_file()
    assert (run/'evidence-assets/generated.png').is_file()
    assert json.loads((run/'evidence.json').read_text())['source']['url'].startswith('fixture:')
