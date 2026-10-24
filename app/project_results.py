"""Present measured reconstruction results without training or promoting failed gates."""
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.artifact_cache import (FileSignature, file_signature, read_json, verified_hash,
                                dataset_fingerprint, clear_verification_cache)
from app.glossary import TERMS
from src.utils.artifacts import resolve_reference
from src.utils.integrity import confined_path
from src.visualization.mri_viewer import TENSOR_SECTIONS, tensor_slice_figure

ARM_NAMES = {
    'ae_vq_baseline': 'VQ baseline · discrete codes',
    'ae_vq_improved': 'VQ · larger codebook',
    'ae_kl_iso': 'KL · isotropic compression',
    'ae_kl_aniso': 'KL · slice-preserving compression',
    'ae_kl_aniso_ssim': 'KL · multi-scale structural loss',
    'ae_kl_aniso_structural': 'KL · structure + edge loss',
    'ae_kl_aniso_perceptual': 'KL · medical feature loss',
    'ae_kl_low_compression': 'KL · higher-detail / no scalar compression',
    'ae_kl_aniso_deconv': 'KL · learned upsampling',
}
MODEL_ORDER = ('ae_kl_aniso', 'ae_kl_low_compression', 'ae_kl_aniso_structural', 'ae_kl_iso',
               'ae_vq_improved', 'ae_vq_baseline', 'ae_kl_aniso_ssim', 'ae_kl_aniso_perceptual', 'ae_kl_aniso_deconv')


@dataclass
class VerifiedRun:
    record: dict[str, Any]
    report: dict[str, Any]
    folder: Path
    checkpoint: Path
    data: Path
    sources: dict[str, dict[str, str]]
    split_counts: Counter


def published_results(root: Path) -> dict[str, Any] | None:
    path = root/'docs/results/ae_bakeoff_measured.json'
    if not path.is_file():
        return None
    value = read_json(path)
    if (not isinstance(value, dict) or not isinstance(value.get('groups'), dict)
            or not value.get('records') or not value.get('dataset_fingerprint')):
        raise ValueError('Published result summary is incomplete')
    if any(r.get('smoke_test', False) for r in value['records']):
        raise ValueError('Artificial runs cannot be presented as project measurements')
    return value


def verified_run(root: Path, record: dict[str, Any], fingerprint: str) -> VerifiedRun:
    study_root = confined_path(root, record['source_study'].replace('\\', '/'))
    study = read_json(study_root/'study.json')
    configs = study.get('plan', {}).get('configs', {})
    research = study.get('smoke_test') is False
    if 'smoke_test' not in study:
        # Early completed studies recorded this flag per arm rather than at root.
        research = bool(configs) and all(c.get('smoke_test') is False for c in configs.values())
    if configs and any(c.get('smoke_test') is not False for c in configs.values()):
        research = False
    if (study.get('status') != 'completed' or not research
            or study['plan']['dataset_fingerprint'] != fingerprint):
        raise ValueError('Only completed matching research studies can supply images')
    matching = next((r for r in study['records'] if r['name'] == record['name']), None)
    if matching is None or any(matching[k] != record[k] for k in ('arm', 'seed', 'checkpoint_sha256')):
        raise ValueError('Published record no longer matches the completed study')
    checkpoint = confined_path(study_root, matching['checkpoint'])
    if verified_hash(checkpoint) != matching['checkpoint_sha256']:
        raise ValueError('Trained checkpoint checksum mismatch')
    folder = confined_path(study_root, f'{matching["name"]}/reconstruction')
    report = read_json(folder/'report.json')
    if (report.get('smoke_test', True) or report['dataset_fingerprint'] != fingerprint
            or report['checkpoint_sha256'] != matching['checkpoint_sha256']):
        raise ValueError('Reconstruction report has stale model or cohort lineage')
    data = resolve_reference(folder, report['input_dataset'])
    if dataset_fingerprint(data) != fingerprint:
        raise ValueError('Reconstruction source data no longer matches its report')
    with (data/'metadata.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    sources = {r['scan_id']: r for r in rows if r['split'] == report['split']}
    if not report['rows'] or any(r['scan_id'] not in sources for r in report['rows']):
        raise ValueError('Reconstruction identities are missing from the declared split')
    return VerifiedRun(matching, report, folder, checkpoint, data, sources, Counter(r['split'] for r in rows))


@st.cache_data(show_spinner=False, max_entries=24)
def _image(signature: FileSignature, checksum: str, shape: tuple[int, ...]) -> np.ndarray:
    path = Path(signature[0])
    if not checksum or verified_hash(path) != checksum:
        raise ValueError('MRI image checksum mismatch')
    array = np.load(path, allow_pickle=False)
    if (array.shape != (1, *shape) or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all() or array.min() < -1.0001 or array.max() > 1.0001):
        raise ValueError('MRI image has invalid shape, values or normalization')
    if file_signature(path) != signature:
        raise ValueError('MRI image changed while loading')
    return array[0]


def image_pair(run: VerifiedRun, identity: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    row = next(r for r in run.report['rows'] if r['scan_id'] == identity)
    source = run.sources[identity]
    shape = tuple(run.report['shape'])
    original = confined_path(run.data, source['output_file'])
    restored = confined_path(run.folder, row['file'])
    return (_image(file_signature(original), source['output_sha256'], shape),
            _image(file_signature(restored), row['sha256'], shape), row)


@st.cache_resource(show_spinner=False, max_entries=2)
def _trained_model(path: str, checksum: str, device: str) -> Any:
    from src.training.representation import load_representation
    if verified_hash(path) != checksum:
        raise ValueError('Live model checksum mismatch')
    return load_representation(path, device)[0]


def live_reconstruction(run: VerifiedRun, original: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    import torch
    from src.models.autoencoder import encode_latent, decode_latent
    from src.training.common import device_for
    from src.metrics import reconstruction_metrics
    torch.set_num_threads(4)
    device = device_for()
    model = _trained_model(str(run.checkpoint), run.record['checkpoint_sha256'], str(device))
    with torch.inference_mode():
        x = torch.from_numpy(original.copy())[None, None].to(device)
        result = decode_latent(model, encode_latent(model, x), project=False)[0, 0].float().cpu().numpy()
    return result, reconstruction_metrics(original, result)


def result_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for arm, group in summary['groups'].items():
        metrics = group['metrics']
        rows.append({'Model': ARM_NAMES.get(arm, arm), 'Seeds': len(group['seeds']),
            'Spatial reduction': ' × '.join(map(str, group['spatial_compression'])),
            'PSNR (dB)': metrics['psnr']['mean'], 'PSNR seed SD': metrics['psnr']['seed_sd'],
            'SSIM': metrics['ssim']['mean'], 'SSIM seed SD': metrics['ssim']['seed_sd'],
            'MS-SSIM': metrics['ms_ssim']['mean'], 'Edge energy ratio': metrics['gradient_energy_ratio']['mean'],
            'HF energy ratio': metrics['high_frequency_energy_ratio']['mean'],
            'Peak training VRAM (MiB)': group['train_peak_mib'],
            'Detail gate': 'Passed' if group['gates_passed'] == len(group['seeds']) else 'Not met'})
    return pd.DataFrame(rows)


def _show_image(array: np.ndarray, section: str, index: int, key: str) -> None:
    figure = tensor_slice_figure(array, section, index)
    figure.update_layout(height=370, margin=dict(l=8, r=8, t=38, b=12))
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)
    st.plotly_chart(figure, width='stretch', key=key, config={'displaylogo': False})


def reconstruction_explorer(root: Path, summary: dict[str, Any]) -> None:
    st.markdown('**Compare the same real MRI across trained models.**')
    st.caption('Reconstruction rebuilds an input scan; generation creates a new one.')
    models = [name for name in MODEL_ORDER if name in summary['groups']]
    controls = st.columns([2.5, 1, 1.8, 1.7])
    arm = controls[0].selectbox('Reconstruction model', models, format_func=lambda name: ARM_NAMES[name],
                         help=TERMS['KL autoencoder'])
    candidates = sorted((r for r in summary['records'] if r['arm'] == arm), key=lambda r: r['seed'])
    seed = controls[1].selectbox('Training seed', [r['seed'] for r in candidates], help=TERMS['Seed'])
    record = next(r for r in candidates if r['seed'] == seed)
    peer = next(r for r in summary['records'] if r['arm'] == 'ae_vq_baseline' and r['seed'] == seed)
    try:
        selected = verified_run(root, record, summary['dataset_fingerprint'])
        baseline = verified_run(root, peer, summary['dataset_fingerprint'])
        ids = [r['scan_id'] for r in selected.report['rows']]
        if set(ids) != {r['scan_id'] for r in baseline.report['rows']} or selected.report['shape'] != baseline.report['shape']:
            raise ValueError('Comparison must use identical validation cases and tensor geometry')
        ranked = sorted(baseline.report['rows'], key=lambda r: r['psnr'])
        preset = controls[2].selectbox('Jump to a case', ['Median baseline case', 'Worst baseline case', 'Best baseline case', 'Seeded random case'])
        identity = {'Median baseline case': ranked[len(ranked)//2]['scan_id'], 'Worst baseline case': ranked[0]['scan_id'],
                    'Best baseline case': ranked[-1]['scan_id'],
                    'Seeded random case': ids[int(np.random.default_rng(42).integers(len(ids)))]}[preset]
        chosen = controls[3].selectbox('Validation exam', range(len(ids)), index=ids.index(identity),
            format_func=lambda i: f'Exam {i+1:02d} of {len(ids)}', key=f'case:{arm}:{seed}:{preset}')
        st.caption(f'{selected.split_counts["train"]} training / {len(ids)} validation exams · model, data and image checksums verified')
        original, reconstruction, metrics = image_pair(selected, ids[chosen])
        _, native, native_metrics = image_pair(baseline, ids[chosen])
        view, slices = st.columns([1, 2])
        section = view.selectbox('View direction', list(TENSOR_SECTIONS), help='Array-grid cross-sections; no patient-space orientation is inferred.')
        axis = TENSOR_SECTIONS[section]
        index = slices.slider('Synchronized MRI slice', 0, original.shape[axis]-1, original.shape[axis]//2,
                          key=f'comparison-slice:{section}:{original.shape}')
        options = st.columns(2)
        show_interpolation = options[0].checkbox('Include interpolation reference', help='Downsample and resize without a learned model.')
        show_errors = options[1].checkbox('Show absolute error maps', help='Brighter pixels indicate larger reconstruction error; the range is fixed across models.')
        arrays = [original, native, reconstruction]
        labels = ['Original MRI', 'Baseline VQ', 'Selected reconstruction']
        captions = ['Real validation input', f'Volume PSNR {native_metrics["psnr"]:.2f} dB · SSIM {native_metrics["ssim"]:.3f}',
                    f'Volume PSNR {metrics["psnr"]:.2f} dB · SSIM {metrics["ssim"]:.3f}']
        if show_interpolation:
            from src.metrics import downsample_reconstruction_baseline
            arrays.append(downsample_reconstruction_baseline(original)[0]); labels.append('Interpolation'); captions.append('Non-learned reference')
        for col, array, label, caption in zip(st.columns(len(arrays)), arrays, labels, captions):
            with col:
                st.markdown(f'**{label}**'); st.caption(caption)
                _show_image(array, section, index, f'image:{label}')
        st.caption('Identical slice positions and fixed [-1,1] display contrast. Metrics describe the whole volume, not just this slice.')
        if arm == 'ae_kl_low_compression':
            st.info('Higher-detail exploratory model: eight latent channels cancel the 2×2×2 spatial reduction, so there is no float-scalar compression.')
        if show_errors:
            for col, array, label in zip(st.columns(2), [native, reconstruction], ['Baseline error', 'Selected-model error']):
                with col:
                    error = np.take(np.abs(array-original), index, axis=axis).T
                    figure = go.Figure(go.Heatmap(z=error, colorscale='Magma', zmin=0, zmax=.5, showscale=True))
                    figure.update_layout(title=label, height=300, margin=dict(l=8, r=8, t=35, b=8))
                    figure.update_yaxes(scaleanchor='x'); st.plotly_chart(figure, width='stretch')
        with st.expander('Run the trained reconstruction now'):
            st.caption('Runs the saved model on this input; no training. The first call loads the model, then later calls reuse it.')
            if st.button('Reconstruct this MRI', type='primary'):
                began = time.perf_counter()
                with st.spinner('Running the trained encoder and decoder…'):
                    result, measured = live_reconstruction(selected, original)
                st.success(f'Inference completed in {time.perf_counter()-began:.2f} s · PSNR {measured["psnr"]:.2f} dB · SSIM {measured["ssim"]:.3f}')
                _show_image(result, section, index, 'live-reconstruction')
        with st.expander('Training curves and artifact details'):
            history = read_json(selected.checkpoint.parent/'history.json')
            frame = pd.DataFrame(history)
            st.line_chart(frame.set_index('epoch')[['l1', 'validation_l1']])
            st.caption('L1 is mean absolute reconstruction error; lower is better. Validation chooses the checkpoint.')
            st.write(dict(model=ARM_NAMES[arm], seed=seed, checkpoint_sha256=record['checkpoint_sha256'],
                input_shape=selected.report['shape'], latent_shape=selected.report['latent_shape'],
                scalar_compression=selected.report['scalar_compression'], quantitative_gate=selected.record['quantitative_gate']))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        st.info('Recorded benchmark results are available. Verified local MRI images/models are required for the interactive comparison.')
        with st.expander('Local artifact details'):
            st.write(f'{type(exc).__name__}: {exc}')


def experiments_panel(summary: dict[str, Any]) -> None:
    st.subheader('Controlled experiments, measured results')
    st.caption('Same cohort and 1,920 optimizer updates per run, effective batch two. Single-seed screens are not three-seed confirmations.')
    frame = result_frame(summary)
    primary = frame[frame['Seeds'] > 1]
    chart = go.Figure(go.Bar(x=primary['Model'], y=primary['PSNR (dB)'],
                            error_y=dict(type='data', array=primary['PSNR seed SD']), marker_color=['#58cbbb', '#7f9ac4', '#acb8c9']))
    chart.update_layout(yaxis_title='Validation PSNR (dB) ↑', height=330, margin=dict(l=15, r=15, t=20, b=15))
    st.plotly_chart(chart, width='stretch')
    st.caption('Error bars are training-seed standard deviations. Higher PSNR means lower pixel error, not verified pathology preservation.')
    display = frame[['Model', 'Seeds', 'Spatial reduction', 'PSNR (dB)', 'SSIM', 'Edge energy ratio', 'HF energy ratio']].copy()
    for key, sd, digits in (('PSNR (dB)', 'PSNR seed SD', 3), ('SSIM', 'SSIM seed SD', 4)):
        display[key] = [f'{mean:.{digits}f} ± {spread:.{digits}f}' if pd.notna(spread) else f'{mean:.{digits}f}'
                        for mean, spread in zip(frame[key], frame[sd])]
    st.dataframe(display, hide_index=True, width='stretch', column_config={
        'Model': st.column_config.TextColumn(width='medium'),
        'Edge energy ratio': st.column_config.NumberColumn('Edge ratio', format='%.3f', help=TERMS['Gradient ratio']),
        'HF energy ratio': st.column_config.NumberColumn('HF ratio', format='%.3f', help=TERMS['High-frequency ratio'])})
    with st.expander('Full technical measurements, memory and seed variance'):
        st.dataframe(frame, hide_index=True, width='stretch', column_config={
        key: st.column_config.NumberColumn(format='%.3f') for key in ('PSNR (dB)', 'PSNR seed SD', 'SSIM', 'SSIM seed SD', 'MS-SSIM', 'Edge energy ratio', 'HF energy ratio')})
    st.caption('Spatial reduction alone is not a codec bitrate. Every tested representation still misses the frozen fine-detail gate.')
    st.download_button('Download experiment table', frame.to_csv(index=False), 'knee-mri-experiments.csv', 'text/csv')
    st.subheader('What is complete?')
    st.dataframe(pd.DataFrame([
        {'Component': 'Reconstruction', 'Measured status': 'Improved across three training seeds; detail gate still not met'},
        {'Component': 'Diffusion generation', 'Measured status': 'Working prototype; four one-MRI sanity runs failed the quality gate'},
        {'Component': 'Pathology conditioning', 'Measured status': 'Numerical conditioning sanity passed; MRI pathology fidelity not established'},
        {'Component': 'Synthetic augmentation', 'Measured status': 'Earlier small probe showed no useful benefit; stronger framework implemented, new serious runs gated'},
    ]), hide_index=True, width='stretch')


def methods_panel(root: Path) -> None:
    st.subheader('How the project works')
    for col, title, detail in zip(st.columns(3), ['01 / Real knee MRI', '02 / Learn a representation', '03 / Reconstruct and evaluate'], [
            'Validate the official exam split, resize to the recorded array grid and normalize intensities.',
            'Compare discrete VQ codes with continuous KL latents; preserve more acquired slices through anisotropic compression.',
            'Decode the scan, compare identical inputs and test whether changes improve structure as well as pixel error.']):
        with col, st.container(border=True):
            st.markdown(f'**{title}**'); st.write(detail)
    st.markdown('**Experimental generation branch:** Gaussian noise → conditioned latent diffusion → trained decoder → quality checks.')
    st.caption('Reconstruction is the demonstrated improvement. Realistic new knee synthesis and reliable pathology control remain research objectives.')
    st.subheader('Technical terms, in one line')
    st.dataframe(pd.DataFrame([{'Term': term, 'Meaning': meaning} for term, meaning in TERMS.items()]),
                 hide_index=True, width='stretch')
    handoff = root/'docs/REPORT_HANDOFF.md'
    if handoff.is_file():
        st.download_button('Download report-writing handoff', handoff.read_text(encoding='utf-8'), 'REPORT_HANDOFF.md', 'text/markdown')


def project_results_page(root: Path) -> None:
    st.write('Real scans. Controlled experiments. Explore the trained models and their measured reconstruction improvements.')
    try:
        summary = published_results(root)
        if summary is None:
            st.info('No completed benchmark summary is present in this workspace. Prepare local study artifacts to explore results.')
            methods_panel(root); return
        base = summary['groups']['ae_vq_baseline']; improved = summary['groups']['ae_kl_aniso']
        bp, kp = base['metrics']['psnr']['mean'], improved['metrics']['psnr']['mean']
        bs, ks = base['metrics']['ssim']['mean'], improved['metrics']['ssim']['mean']
        cols = st.columns(4)
        cols[0].metric('Baseline VQ · PSNR', f'{bp:.2f} dB', help=TERMS['PSNR'])
        cols[1].metric('Anisotropic KL · PSNR', f'{kp:.2f} dB', f'{kp-bp:+.2f} dB', help=TERMS['Anisotropic compression'])
        cols[2].metric('Anisotropic KL · SSIM', f'{ks:.3f}', f'{ks-bs:+.3f}', help=TERMS['SSIM'])
        cols[3].metric('Completed training runs', len(summary['records']), help=TERMS['Seed'])
        st.caption(f'Means over {len(base["seeds"])} baseline / {len(improved["seeds"])} KL seeds. '
                   +'PSNR measures pixel error; SSIM measures structural similarity. These are model-selection validation results.')
        compare, experiments, methods = st.tabs(['MRI comparison', 'Experiment results', 'Method & plain-language guide'])
        with compare:
            reconstruction_explorer(root, summary)
        with experiments:
            experiments_panel(summary)
        with methods:
            methods_panel(root)
        with st.expander('Recheck saved results'):
            st.caption('Checksums link the displayed MRI to its model, dataset and completed experiment.')
            if st.button('Reverify project artifacts'):
                clear_verification_cache(); _image.clear(); _trained_model.clear(); st.rerun()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'Project results could not be verified: {type(exc).__name__}. Check the saved summary and local study artifacts.')
