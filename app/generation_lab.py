"""A coherent generation workspace with model-bound samples and explicit research status."""
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any

import streamlit as st

from app.artifact_cache import read_json, verified_hash, file_signature
from app.project_results import _image
from app.research_workspace import research_runs, interactive_samples, interactive_manifest
from app.ui import page_header, viewport, note, stat_cards
from app.glossary import TERMS
from src.utils.integrity import confined_path
from src.visualization.mri_viewer import TENSOR_SECTIONS


def generator_artifacts(run: Path) -> tuple[dict[str, Any], Path, Path]:
    paths = []
    for stage in ('vq', 'diffusion'):
        info = read_json(run/stage/'run.json')
        path = confined_path(run/stage, info.get('checkpoint_file', 'last.pt'))
        if info.get('smoke_test', True) or info.get('completed_epochs', 0) < 1 or verified_hash(path) != info.get('checkpoint_sha256'):
            raise ValueError('Generator stages must be completed and hash-verified')
        paths.append(path)
    manifest = read_json(run/'generated/manifest.json')
    if (manifest.get('smoke_test', True) or not manifest.get('rows') or
            manifest['vq_sha256'] != verified_hash(paths[0]) or manifest['diffusion_sha256'] != verified_hash(paths[1])):
        raise ValueError('Saved samples do not match the selected trained model pair')
    return manifest, paths[0], paths[1]


def generation_page(root: Path) -> None:
    page_header('Latent diffusion', 'Generation lab',
        'Explore saved model outputs and generate a volume using selected seeds, samplers and conditioning inputs.',
        '3D sampling')
    with st.expander('Model connection & artifact checks'):
        directory = Path(st.text_input('Research runs directory', str(root/'runs')))
        if st.button('Reverify generator artifacts'):
            from app.artifact_cache import clear_verification_cache
            clear_verification_cache()
    runs = research_runs(directory)
    if not runs:
        note('Connect a completed model run above to browse samples and generate a volume.')
        return
    try:
        controls = st.columns([1.5, 2, 1.5])
        run = controls[0].selectbox('Trained generator', runs, format_func=lambda path: path.name)
        manifest, vq, diffusion = generator_artifacts(run)
        recent, rejected = interactive_samples(run, manifest)
        latest = st.session_state.get('generation_latest')
        choices: list[tuple[Path, dict[str, Any], str]] = []
        if latest and latest['run'] == str(run):
            folder = Path(latest['folder'])
            info = interactive_manifest(folder, manifest)
            choices.append((folder, info['rows'][0], f'Latest live request · seed {info["rows"][0]["seed"]}'))
        choices.extend((run/'generated', row, f'Saved sample {i+1:02d} · seed {row["seed"]}') for i, row in enumerate(manifest['rows']))
        for folder in recent[:12]:
            if not choices or folder != choices[0][0]:
                info = interactive_manifest(folder, manifest)
                choices.append((folder, info['rows'][0], f'Previous live request · seed {info["rows"][0]["seed"]} · {folder.name[-6:]}'))
        selection = controls[1].selectbox('Diffusion sample', range(len(choices)), format_func=lambda i: choices[i][2],
            key=f'generator_sample:{manifest["diffusion_sha256"]}:{choices[0][0]}')
        section = controls[2].selectbox('Sample view direction', list(TENSOR_SECTIONS))
        folder, row, _ = choices[selection]
        path = confined_path(folder, row['file'])
        array = _image(file_signature(path), row['sha256'], tuple(manifest['shape']))
        axis = TENSOR_SECTIONS[section]
        frame = st.slider('Generated slice', 1, array.shape[axis], array.shape[axis]//2+1, key=f'generated_frame:{section}:{array.shape}')
        image, context = st.columns([2.1, 1], gap='large')
        with image:
            viewport(array, section, frame-1, 'Diffusion output', f'Seed {row["seed"]} · slice {frame}/{array.shape[axis]} · verified saved output', 'generation_preview', 440)
        with context:
            st.markdown('### What you are seeing')
            st.write(TERMS['Generation'])
            condition = row.get('conditioning_labels')
            st.markdown('**Requested condition**')
            if condition is not None:
                for task, value in zip(('Abnormal', 'ACL', 'Meniscus'), condition):
                    st.write(f'{task}: {"positive" if value else "negative"}')
            else:
                st.write('Unconditional sample')
            st.caption('The displayed values are the global label inputs used for this sample.')
            st.download_button('Download generated tensor', path.read_bytes(), path.name, width='stretch')
        with st.expander('Generate a new volume'):
            st.caption('A real inference request writes one fresh, traceable sample. It does not retrain the model.')
            with st.form('generation_request'):
                fields = st.columns(3)
                seed = int(fields[0].number_input('Sampling seed', min_value=0, max_value=2147483647, value=2026))
                method = fields[1].selectbox('Sampling method', ['ddim', 'ddpm'], help=TERMS['DDIM'])
                training_steps = int(manifest.get('sampling', {}).get('training_steps', 200))
                steps = int(fields[2].number_input('DDIM sampling steps', 1, training_steps, min(32, training_steps)))
                default = manifest['rows'][0].get('conditioning_labels')
                labels = [int(col.checkbox(task, value=bool(value))) for col, task, value in
                          zip(st.columns(3), ('Abnormal', 'ACL', 'Meniscus'), default)] if default is not None else None
                guided = manifest.get('sampling', {}).get('classifier_free_guidance_trained', False)
                guidance = float(st.number_input('Guidance scale', 0., 5., 1., .1)) if guided else 1.
                submitted = st.form_submit_button('Generate one volume', type='primary')
            if submitted:
                from src.training.experiments import generate
                target = run/'interactive'/f'sample-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}'
                with st.spinner('Denoising latent noise with the selected trained model…'):
                    generate(vq, diffusion, target, count=1, seed=seed, labels=labels,
                             sampler=method, sampling_steps=steps if method == 'ddim' else None, guidance_scale=guidance)
                st.session_state['generation_latest'] = dict(run=str(run), folder=str(target)); st.rerun()
        with st.expander('Training history & model details'):
            stat_cards([('Saved research samples', str(len(manifest['rows'])), 'Original completed experiment'),
                        ('Training schedule', str(manifest.get('sampling', {}).get('training_steps', 'Recorded')), 'Denoising timesteps'),
                        ('Volume shape', ' × '.join(map(str, manifest['shape'])), 'Recorded output grid'),
                        ('Conditioning', 'Global labels', 'Abnormal · ACL · meniscus')])
            if rejected:
                st.caption(f'{len(rejected)} incompatible or incomplete interactive folders were excluded.')
            columns = st.columns(2)
            import pandas as pd
            for col, stage, key in zip(columns, ('vq', 'diffusion'), ('reconstruction_l1', 'noise_mse')):
                history = run/stage/'history.json'
                if history.is_file():
                    with col:
                        st.markdown(f'**{stage.upper()} training**')
                        st.line_chart(pd.DataFrame(read_json(history)).set_index('epoch')[[key]])
            report = run/'generation-report.json'
            if report.is_file():
                st.download_button('Download original generation diagnostics', report.read_bytes(), 'generation-report.json')
            st.json(dict(vq_sha256=manifest['vq_sha256'], diffusion_sha256=manifest['diffusion_sha256'],
                         tensor_shape=manifest['shape'], sampling=manifest.get('sampling', {})), expanded=False)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        st.error(f'The selected generation artifacts could not be verified ({type(exc).__name__}). Choose a completed matching run or recheck its files.')
