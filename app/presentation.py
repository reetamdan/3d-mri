"""Research workspace pages backed by recorded study artifacts."""
from html import escape
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.artifact_cache import read_json, clear_verification_cache
from app.glossary import TERMS
from app.project_results import (published_results, verified_run, image_pair, live_reconstruction,
                                result_frame, ARM_NAMES, MODEL_ORDER, VerifiedRun)
from app.ui import page_header, stat_cards, note, preview_pair, viewport, comparison_figure
from src.visualization.mri_viewer import TENSOR_SECTIONS

SHORT_NAMES = {'ae_vq_baseline': 'VQ baseline', 'ae_kl_iso': 'KL isotropic', 'ae_kl_aniso': 'KL anisotropic'}


def project_summary(root: Path) -> dict[str, Any] | None:
    try:
        return published_results(root)
    except (OSError, ValueError, KeyError, TypeError):
        st.error('The recorded study summary could not be verified. Restore the completed summary to continue.')
        return None


def run_pair(root: Path, summary: dict[str, Any], arm: str, seed: int) -> tuple[VerifiedRun, VerifiedRun]:
    record = next(r for r in summary['records'] if r['arm'] == arm and r['seed'] == seed)
    peer = next(r for r in summary['records'] if r['arm'] == 'ae_vq_baseline' and r['seed'] == seed)
    chosen, baseline = (verified_run(root, row, summary['dataset_fingerprint']) for row in (record, peer))
    if (chosen.report['shape'] != baseline.report['shape'] or
            {r['scan_id'] for r in chosen.report['rows']} != {r['scan_id'] for r in baseline.report['rows']}):
        raise ValueError('Comparisons require identical cases and geometry')
    return chosen, baseline


def _local_notice(exc: Exception) -> None:
    note('Recorded results are available. Interactive MRI needs the matching local dataset, checkpoints and saved reconstructions.')
    with st.expander('Connect local study artifacts'):
        st.write('Connect the matching data/ and runs/ folders, or run a representation study. Images are loaded from local files.')
        st.code('python scripts/representation_study.py --output runs/AE-BAKEOFF-new --device cuda', language='bash')
        st.caption(f'Artifact check: {type(exc).__name__}. Check the local dataset and run locations.')
        st.code(str(exc), language='text')


def overview_page(root: Path) -> None:
    summary = project_summary(root)
    with st.container(key='overview_hero'):
        intro, image = st.columns([1.1, 1], gap='large')
        with intro:
            page_header('The project', '3D Knee MRI',
                'From real scans to learned reconstruction. A controlled study of what a compact MRI representation can preserve.',
                'Measured study', 'neutral')
            st.write('**Learned representations. Recorded experiments. Interactive MRI tools.**')
            st.caption('Compare the original MRI with a discrete-code VQ baseline and a continuous KL model that retains more acquired slices.')
            st.page_link('pages/reconstruction.py', label='Open reconstruction studio', icon=':material/arrow_forward:')
            st.page_link('pages/experiments.py', label='Explore the measured results', icon=':material/monitoring:')
        with image:
            if summary:
                try:
                    selected, baseline = run_pair(root, summary, 'ae_kl_aniso', 42)
                    ranked = sorted(baseline.report['rows'], key=lambda r: r['psnr'])
                    identity = ranked[len(ranked)//2]['scan_id']
                    original, restored, _ = image_pair(selected, identity)
                    preview_pair(original, restored, original.shape[2]//2)
                    st.caption('A baseline-median validation case · same slice · fixed contrast')
                except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
                    _local_notice(exc)
            else:
                note('The project is ready for a local completed study. Add its recorded result summary and artifacts to inspect MRI.')
    if summary:
        base = summary['groups']['ae_vq_baseline']['metrics']; improved = summary['groups']['ae_kl_aniso']['metrics']
        stat_cards([
            ('KL reconstruction · PSNR', f'{improved["psnr"]["mean"]:.2f} dB', f'+{improved["psnr"]["mean"]-base["psnr"]["mean"]:.2f} dB versus matched VQ'),
            ('Structural similarity · SSIM', f'{improved["ssim"]["mean"]:.3f}', 'Three training seeds · internal validation'),
            ('Controlled training runs', str(len(summary['records'])), f'{len(summary["groups"])} representation variants'),
            ('Comparison protocol', 'Same data', 'Matched updates, batch size and evaluation')])
    st.subheader('Follow the study')
    routes = [
        ('01 / EXPLORE', 'Real knee MRI', 'Inspect the acquired series, exam-level labels and frozen cohort.', 'pages/dataset.py', 'Open knee dataset'),
        ('02 / COMPARE', 'Learned reconstruction', 'Use linked slices, shared contrast and error maps to examine real differences.', 'pages/reconstruction.py', 'Open MRI studio'),
        ('03 / GENERATE', 'Latent diffusion', 'Explore saved samples and run the trained generator with selected inputs.', 'pages/generation.py', 'Open generation lab')]
    for col, (step, title, text, path, label) in zip(st.columns(3), routes):
        with col:
            st.markdown(f'<div class="lab-route-card"><small>{escape(step)}</small><h3>{escape(title)}</h3><p>{escape(text)}</p></div>', unsafe_allow_html=True)
            st.page_link(path, label=label, icon=':material/arrow_forward:')


def _set_exam(identity: str) -> None:
    st.session_state['studio_exam'] = identity


def studio_page(root: Path) -> None:
    page_header('Image comparison', 'Reconstruction studio',
        'Same real exam. Linked slices. Shared display contrast. Start with two large viewports, then add references.',
        'Learned representation')
    summary = project_summary(root)
    if not summary:
        note('A completed representation summary is needed to open the studio.'); return
    models = [arm for arm in MODEL_ORDER if arm in summary['groups']]
    preferences = st.session_state.setdefault('studio_preferences', {})
    preferred_model = preferences.get('model', models[0])
    with st.container(key='studio_toolbar'):
        model_col, exam_col, direction_col = st.columns([2.2, 1.6, 1.7])
        arm = model_col.selectbox('Reconstruction model', models, index=models.index(preferred_model) if preferred_model in models else 0,
            format_func=lambda name: ARM_NAMES[name], help=TERMS['KL autoencoder'], key='studio_model')
    with st.expander('Display, training seed & comparison options'):
        options = st.columns(4)
        seeds = sorted(r['seed'] for r in summary['records'] if r['arm'] == arm)
        preferred_seed = preferences.get('seeds', {}).get(arm, seeds[0])
        seed = options[0].selectbox('Training seed', seeds, index=seeds.index(preferred_seed) if preferred_seed in seeds else 0,
                                   key=f'studio_seed:{arm}', help=TERMS['Seed'])
        height = options[1].selectbox('Image height', [320, 420, 560], index=[320, 420, 560].index(preferences.get('height', 320)),
                                     format_func=lambda n: f'{n} px', key='studio_height')
        show_vq = options[2].checkbox('Include VQ baseline', value=preferences.get('vq', False), key='studio_include_vq')
        show_interpolation = options[2].checkbox('Include interpolation reference', value=preferences.get('interpolation', False), key='studio_include_interpolation')
        show_errors = options[3].checkbox('Show absolute error maps', value=preferences.get('errors', False), key='studio_errors')
        limits = st.slider('Shared display window', -1., 1., tuple(preferences.get('window', (-1., 1.))), .05, key='studio_window',
            help='Display contrast only. The same window is applied to every panel; metrics and model arrays are unchanged.')
        if limits[1] <= limits[0]:
            st.info('Choose a display window with a nonzero width.'); return
    preferences.update(model=arm, height=height, vq=show_vq, interpolation=show_interpolation, errors=show_errors, window=limits)
    preferences.setdefault('seeds', {})[arm] = seed
    canvas = st.container(key='studio_canvas')
    try:
        selected, baseline = run_pair(root, summary, arm, seed)
        rows = selected.report['rows']; ids = [r['scan_id'] for r in rows]
        ranked = sorted(baseline.report['rows'], key=lambda r: r['psnr'])
        default = preferences.get('exam', ranked[len(ranked)//2]['scan_id'])
        if default not in ids:
            default = ranked[len(ranked)//2]['scan_id']
        identity = exam_col.selectbox('Validation exam', ids, index=ids.index(default), key='studio_exam',
            format_func=lambda value: f'Exam {ids.index(value)+1:02d} / {len(ids)}')
        sections = list(TENSOR_SECTIONS)
        section = direction_col.selectbox('View direction', sections, index=sections.index(preferences.get('view', sections[0])), key='studio_view',
            help='Array-grid views; physical orientation and spacing are unavailable in these MRNet tensors.')
        preferences.update(exam=identity, view=section)
        original, saved, metrics = image_pair(selected, identity)
        _, native, native_metrics = image_pair(baseline, identity)
        axis = TENSOR_SECTIONS[section]
        with canvas.container(key='view_controls'):
            slider_col, action = st.columns([4, 1])
            frame = slider_col.slider('Synchronized MRI slice', 1, original.shape[axis], original.shape[axis]//2+1,
                                      key=f'studio_slice:{section}:{original.shape}')
            action.caption('Saved model · no training')
            run_now = action.button('Run reconstruction', type='primary', width='stretch')
        token = (selected.record['checkpoint_sha256'], identity)
        if run_now:
            with st.spinner('Loading the trained model and reconstructing this scan…'):
                began = time.perf_counter()
                try:
                    result, measured = live_reconstruction(selected, original)
                    st.session_state['studio_live'] = dict(token=token, array=result, metrics=measured, seconds=time.perf_counter()-began)
                except (RuntimeError, OSError, ValueError) as exc:
                    canvas.error(f'Live inference could not complete ({type(exc).__name__}). The verified saved comparison is still available.')
        live = st.session_state.get('studio_live')
        active_live = live if live and live['token'] == token else None
        restored = active_live['array'] if active_live else saved
        measured = active_live['metrics'] if active_live else metrics
        arrays, names, details = [original, restored], ['Original MRI', 'KL reconstruction' if arm.startswith('ae_kl') else 'VQ reconstruction'], [
            'Real validation input', f'Whole-volume PSNR {measured["psnr"]:.2f} dB · SSIM {measured["ssim"]:.3f}']
        if show_vq:
            arrays.insert(1, native); names.insert(1, 'VQ baseline'); details.insert(1, f'Whole-volume PSNR {native_metrics["psnr"]:.2f} dB · SSIM {native_metrics["ssim"]:.3f}')
        if show_interpolation:
            from src.metrics import downsample_reconstruction_baseline
            arrays.append(downsample_reconstruction_baseline(original)[0]); names.append('Interpolation'); details.append('Non-learned downsample / upsample reference')
        with canvas.container(key='image_grid'):
            for col, array, name, detail in zip(st.columns(len(arrays)), arrays, names, details):
                with col:
                    viewport(array, section, frame-1, name, detail, f'studio_{name}', height, limits)
        canvas.caption(f'{selected.split_counts["train"]} training / {len(ids)} model-selection validation exams · '
                   f'slice {frame}/{original.shape[axis]} · model, data and image hashes verified')
        if active_live:
            canvas.success(f'Live reconstruction completed in {active_live["seconds"]:.2f} s. The live volume stays available while you move between slices.')
        if arm == 'ae_kl_low_compression':
            with canvas:
                note('Higher-detail configuration: 2×2×2 spatial reduction with eight continuous latent channels.')
        if show_errors:
            with canvas.container(key='error_grid'):
                for col, array, label in zip(st.columns(2), [native, restored], ['VQ absolute error', 'Selected-model absolute error']):
                    with col:
                        error = np.take(np.abs(array-original), frame-1, axis=axis).T
                        fig = go.Figure(go.Heatmap(z=error, colorscale='Magma', zmin=0, zmax=.5, showscale=True))
                        viewport(array, section, frame-1, label, 'Fixed error scale 0–0.5 · brighter means larger error', label, 300, figure=fig)
        export = comparison_figure(arrays, names, section, frame-1, limits,
            f'Exam {ids.index(identity)+1:02d} | {section} {frame} | display window {limits} | model {selected.record["checkpoint_sha256"][:12]}')
        download, annotation = canvas.columns([1, 3])
        download.download_button('Download comparison figure', export, f'reconstruction-exam-{ids.index(identity)+1:02d}-slice-{frame}.png', 'image/png', width='stretch')
        annotation.caption('The panels share one slice and display window. PSNR and SSIM describe the whole reconstructed volume.')
        with canvas.expander('Representative cases, training curves & provenance'):
            shortcuts = st.columns(4)
            targets = [ranked[len(ranked)//2]['scan_id'], ranked[0]['scan_id'], ranked[-1]['scan_id'], ids[int(np.random.default_rng(42).integers(len(ids)))]]
            for col, label, target in zip(shortcuts, ['Median case', 'Lower-PSNR case', 'Higher-PSNR case', 'Seeded random case'], targets):
                col.button(label, on_click=_set_exam, args=(target,), width='stretch')
            history = pd.DataFrame(read_json(selected.checkpoint.parent/'history.json'))
            st.line_chart(history.set_index('epoch')[['l1', 'validation_l1']])
            st.caption('L1 is mean absolute reconstruction error; lower is better. The validation set selected the checkpoint.')
            st.json(dict(seed=seed, input_shape=selected.report['shape'], latent_shape=selected.report['latent_shape'],
                scalar_compression=selected.report['scalar_compression'], checkpoint_sha256=selected.record['checkpoint_sha256']), expanded=False)
        canvas.caption('Display windows and zoom affect the viewer only; saved arrays and recorded measurements are unchanged.')
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        _local_notice(exc)


def experiment_page(root: Path) -> None:
    page_header('Measurements', 'Experiment results', 'Compare representation settings and recorded reconstruction measurements.', 'Research study')
    summary = project_summary(root)
    if not summary:
        note('Add a completed study summary to inspect measured results.'); return
    groups = summary['groups']
    primary = [arm for arm in ('ae_vq_baseline', 'ae_kl_iso', 'ae_kl_aniso') if arm in groups]
    psnr = groups['ae_kl_aniso']['metrics']['psnr']['mean']-groups['ae_vq_baseline']['metrics']['psnr']['mean']
    ssim = groups['ae_kl_aniso']['metrics']['ssim']['mean']-groups['ae_vq_baseline']['metrics']['ssim']['mean']
    stat_cards([('PSNR gain', f'+{psnr:.2f} dB', 'Anisotropic KL versus matched native VQ'),
                ('SSIM gain', f'+{ssim:.4f}', 'Across three actual training seeds'),
                ('Optimization budget', '1,920', 'Updates per run · effective batch two'),
                ('Representation variants', str(len(groups)), 'Sampling and reconstruction-loss configurations')])
    for col, key, title in zip(st.columns(2), ['psnr', 'ssim'], ['Pixel accuracy · PSNR (dB)', 'Structural similarity · SSIM']):
        with col:
            st.markdown(f'**{title}**')
            fig = go.Figure(go.Scatter(x=[groups[a]['metrics'][key]['mean'] for a in primary],
                y=[SHORT_NAMES[a] for a in primary], mode='markers+text',
                text=[f'{groups[a]["metrics"][key]["mean"]:.3f}' for a in primary], textposition='top center',
                error_x=dict(type='data', array=[groups[a]['metrics'][key]['seed_sd'] or 0 for a in primary]),
                marker=dict(size=13, color=['#91a8c5', '#8ba4ed', '#62e0c2'])))
            fig.update_layout(height=260, margin=dict(l=10, r=35, t=25, b=25), xaxis_title=title,
                yaxis=dict(autorange='reversed'), paper_bgcolor='#0a1220', plot_bgcolor='#0a1220')
            st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
    st.caption('Points are validation means; error bars show variation across training seeds. The recorded study uses 96 training and 24 validation exams.')
    frame = result_frame(summary)
    show_all = st.toggle('Include exploratory one-seed variants', value=False)
    visible = frame if show_all else frame[frame['Seeds'] > 1]
    columns = ['Model', 'Seeds', 'Spatial reduction', 'PSNR (dB)', 'SSIM', 'Edge energy ratio', 'HF energy ratio']
    st.dataframe(visible[columns], hide_index=True, width='stretch', column_config={
        'Model': st.column_config.TextColumn(width='medium'),
        **{k: st.column_config.NumberColumn(format='%.4f') for k in columns[3:]}})
    with st.expander('Full measurements and memory'):
        st.dataframe(frame, hide_index=True, width='stretch')
    st.download_button('Download full results CSV', frame.to_csv(index=False), 'knee-mri-experiments.csv', 'text/csv')


def about_page(root: Path) -> None:
    page_header('How it works', 'About & terminology', 'A short guide to the MRI workflow and the terms used in the workspace.', '3D MRI research')
    st.subheader('From MRI to measured evidence')
    steps = [('01', 'Real MRI', 'Official exam split, validated arrays and recorded normalization.'),
             ('02', 'Compact representation', 'VQ codes or continuous KL latents, including slice-preserving sampling.'),
             ('03', 'Reconstruct or generate', 'Reconstruct input MRI; separately investigate denoising new latent noise.'),
             ('04', 'Measure outputs', 'Compare reconstruction, structure, edges and sample statistics.')]
    st.markdown('<div class="lab-pipeline">'+''.join(f'<div class="lab-step"><small>{n}</small><b>{title}</b><small>{text}</small></div>' for n, title, text in steps)+'</div>', unsafe_allow_html=True)
    st.subheader('Technical terms, in one line')
    search = st.text_input('Find a term', placeholder='Try: latent, SSIM, conditioning…')
    all_terms = st.toggle('Show full glossary', value=False)
    essentials = ('Reconstruction', 'Generation', 'Latent representation', 'KL autoencoder', 'Anisotropic compression', 'PSNR', 'SSIM', 'Conditioning')
    matches = [(term, meaning) for term, meaning in TERMS.items()
               if search.casefold() in (term+' '+meaning).casefold() and (search or all_terms or term in essentials)]
    if not matches:
        st.info('No matching term. Try a shorter word.')
    for col, values in zip(st.columns(2), (matches[::2], matches[1::2])):
        with col:
            for term, meaning in values:
                with st.container(border=True):
                    st.markdown(f'**{term}**'); st.caption(meaning)
    with st.expander('Reproducibility and artifact checks'):
        st.write('The workspace links displayed images to saved models, datasets and experiments using content hashes. Data and weights are stored locally.')
        if st.button('Clear verification caches'):
            from app.project_results import _image, _trained_model
            from app.dataset_explorer import _series
            clear_verification_cache(); _image.clear(); _trained_model.clear(); _series.clear()
            st.success('Cached model, image and content checks cleared. Files will be rechecked on the next image view.')
