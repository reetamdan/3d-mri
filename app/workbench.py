"""Existing NIfTI pipeline tools, organized as an optional, self-contained workspace."""
import io
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from app.ui import page_header, note, viewport, stat_cards
from src.utils.config import load_config, validate_config
from src.data.pipeline import inspect_dataset, run_pipeline
from src.data.loader import load_mri
from src.data.preprocessing import preprocess_volume
from src.data.splitting import split_patients
from src.visualization.mri_viewer import PLANES, canonical_volume, slice_figure, aligned_original
from src.visualization.plots import histogram, intensity_stats
from src.visualization.volume_viewer import volume_figure
from src.metrics import downsample_reconstruction_baseline

TOOLS = ['Dataset Overview', 'MRI Viewer', 'Preprocessing', 'Intensity Analysis', 'Dataset Quality', 'Export', 'Legacy model audit']


def _remember(key: str) -> None:
    st.session_state[key] = st.session_state['_'+key]
    if key == 'nifti_config':
        st.session_state.pop('nifti_raw', None); st.session_state.pop('_nifti_raw', None)


def _connection(root: Path) -> dict[str, Any] | None:
    with st.expander('Dataset connection & processing configuration'):
        default = str(root/'configs/config.yaml') if (root/'configs/config.yaml').is_file() else ''
        path = st.text_input('Configuration file', st.session_state.get('nifti_config', default),
                             key='_nifti_config', on_change=_remember, args=('nifti_config',),
                             help='Leave blank to use the bundled preprocessing preset.')
        try:
            config = load_config(path)
        except Exception as exc:
            st.error(f'Configuration could not be loaded: {type(exc).__name__}. Check the YAML and paths.'); return None
        raw = st.text_input('Raw dataset directory', st.session_state.get('nifti_raw', config['dataset']['raw_dir']),
                           key='_nifti_raw', on_change=_remember, args=('nifti_raw',))
        config['dataset']['raw_dir'] = str(Path(raw).resolve())
        st.caption('NIfTI inputs carry physical geometry. The local Slicer brain examples test this pipeline; they are separate from the MRNet knee study.')
        refresh = st.button('Index & validate dataset', type='primary')
    fingerprint = json.dumps(config, sort_keys=True)
    if st.session_state.get('dataset_key') != fingerprint:
        for key in ('records', 'errors', 'report', 'preview', 'run'):
            st.session_state.pop(key, None)
        st.session_state.dataset_key = fingerprint
    auto = st.session_state.get('nifti_index_attempt') != fingerprint and Path(config['dataset']['raw_dir']).is_dir()
    if refresh or auto:
        for key in ('records', 'errors', 'report', 'preview', 'run'):
            st.session_state.pop(key, None)
        st.session_state['nifti_index_attempt'] = fingerprint
        try:
            with st.spinner('Validating local NIfTI scans…'):
                records, errors, report = inspect_dataset(config)
            st.session_state.update(records=records, errors=errors, report=report)
        except Exception as exc:
            st.error(str(exc) if isinstance(exc, ValueError) else f'Dataset indexing could not complete: {type(exc).__name__}')
    return config


def workbench_page(root: Path) -> None:
    page_header('Additional research tools', 'Data workbench',
        'Inspect and process geometry-aware NIfTI data, or audit the original model baseline. These tools do not change the frozen knee study.', 'Advanced tools')
    page = st.selectbox('Data tool', TOOLS, index=TOOLS.index(st.session_state.get('last_data_tool', TOOLS[0])), key='data_tool')
    st.session_state['last_data_tool'] = page
    if page == 'Legacy model audit':
        from app.research_workspace import experiments_page
        experiments_page(root)
        return
    config = _connection(root)
    if config is None:
        return
    records = st.session_state.get('records', [])
    report = st.session_state.get('report')
    if report is None:
        note('Connect a local NIfTI dataset above, then index it. The knee image tools are available in their respective pages.')
        with st.expander('Expected data layout'):
            st.code('data/raw/\n  subject_001/scan.nii.gz\n  subject_002/scan.nii.gz', language='text')
            st.caption('Flat folders need an explicit file_path,patient_id CSV manifest. No data is downloaded automatically.')
        return
    if page == 'Dataset Overview':
        splits = split_patients([r['patient_id'] for r in records], config['split'], config['seed'])
        cols = st.columns(4)
        cols[0].metric('Valid scans', len(records))
        for col, split in zip(cols[1:], ('train', 'validation', 'test')):
            col.metric(f'{split.title()} · planned', sum(splits[r['patient_id']] == split for r in records))
        st.caption('Planned splits group the currently valid subjects. Final export counts and any rejected scans are recorded in the run report.')
        if records:
            st.subheader('Geometry at a glance')
            st.dataframe(pd.DataFrame({'Statistic': ['Minimum', 'Mean', 'Maximum'],
                'Source shape': [' × '.join(f'{n:g}' for n in report['shape'][k]) for k in ('min', 'mean', 'max')],
                'Spacing · native units': [' × '.join(f'{n:.3f}' for n in report['spacing_native_units'][k]) for k in ('min', 'mean', 'max')]}), hide_index=True, width='stretch')
            units = ', '.join(f'{unit} ({count} scans)' for unit, count in report['spatial_units'].items())
            st.caption(f'{report["subjects"]} subjects · {report["distinct_shapes"]} shapes · {units} · {report["invalid_scans"]} invalid scans')
        note('Next: choose MRI Viewer to inspect a scan, Preprocessing for an aligned preview, or Export for a new processed run.')
    elif page == 'Dataset Quality':
        for col, key in zip(st.columns(3), ('discovered_scans', 'valid_scans', 'invalid_scans')):
            col.metric(key.replace('_', ' ').title(), report[key])
        if st.session_state.get('errors'):
            st.dataframe(st.session_state.errors, hide_index=True, width='stretch')
        else:
            st.success('No invalid scans in the current index.')
        with st.expander('Full quality measurements'):
            st.json(report)
        st.download_button('Download quality report', json.dumps(report, indent=2), 'quality_report.json', 'application/json')
    elif page == 'Export':
        st.subheader('Create a processed dataset')
        stat_cards([('Valid inputs', str(len(records)), 'Current indexed scans'),
                    ('Output format', config.get('export_format', 'npy').upper(), 'Saved arrays and metadata'),
                    ('Target shape', ' × '.join(map(str, config['preprocessing']['target_shape'])), 'One channel plus three spatial axes'),
                    ('Split seed', str(config['seed']), 'Subject-safe deterministic assignments')])
        st.caption('Export uses the saved configuration, not temporary preview controls. Each request creates a fresh output directory.')
        with st.expander('Export settings and destination'):
            st.code(yaml.safe_dump(config['preprocessing']), language='yaml')
            st.code(config['dataset']['processed_dir'], language='text')
        if st.button('Process & export dataset', type='primary', disabled=not records):
            bar = st.progress(0.)
            try:
                st.session_state.run = run_pipeline(config, progress=bar.progress)
            except Exception as exc:
                st.error(f'Export could not complete: {type(exc).__name__}. Check the output directory and configuration.')
        if st.session_state.get('run'):
            run = st.session_state.run
            (st.success if run['rows'] else st.error)(f'Exported {len(run["rows"])} scans; {len(run["errors"])} rejected.')
            with st.expander('Completed export details'):
                st.code(run['output_dir'], language='text'); st.json(run['report'])
            for col, name in zip(st.columns(3), ('metadata.csv', 'report.json', 'validation_errors.csv')):
                col.download_button(f'Download {name}', (Path(run['output_dir'])/name).read_bytes(), name, width='stretch')
    else:
        if not records:
            st.warning('No valid scans are available. Review Dataset Quality or change the connection.'); return
        selected = st.selectbox('Scan', [r['scan_id'] for r in records], format_func=lambda value: f'Scan {next(i for i,r in enumerate(records) if r["scan_id"] == value)+1:02d}')
        record = next(r for r in records if r['scan_id'] == selected)
        try:
            volume, meta = load_mri(record['path'], config['preprocessing'].get('max_voxels', 67108864))
        except Exception:
            st.error('This scan is no longer readable. Re-index the dataset.'); return
        canonical, affine = canonical_volume(volume, meta['affine'])
        if page == 'MRI Viewer':
            with st.container(key='image_grid'):
                for col, plane in zip(st.columns(3), PLANES):
                    with col:
                        count = canonical.shape[PLANES[plane]]
                        index = st.slider(f'{plane} slice', 0, count-1, count//2) if count > 1 else 0
                        fig = slice_figure(canonical, plane, index, nib.affines.voxel_sizes(affine))
                        viewport(canonical, 'Acquired-series slice', 0, plane, f'Canonical RAS · slice {index+1}/{count}', f'nifti_{plane}', 330, figure=fig)
            st.caption(f'Source shape {meta["volume_shape"]} · source orientation {meta["orientation"]} · display canonical RAS · units {meta["spatial_unit"]}')
            with st.expander('Spatial metadata'):
                st.json({key: value for key, value in meta.items() if key != 'file_path'})
            if st.checkbox('Render 3D volume (downsampled for performance)'):
                st.plotly_chart(volume_figure(canonical, affine), width='stretch')
        else:
            p = dict(config['preprocessing'])
            if page == 'Preprocessing':
                with st.expander('Preview parameters', expanded=False):
                    st.caption('Preview settings do not change batch export; download the preview YAML to reuse them.')
                    p['target_shape'] = [int(col.number_input(f'Target {label}', min_value=1, max_value=max(1024, int(value)), value=int(value)))
                        for col, label, value in zip(st.columns(3), ('R', 'A', 'S'), p['target_shape'])]
                    p['normalization'] = st.selectbox('Normalization', ['minmax', 'zscore'], index=['minmax', 'zscore'].index(p['normalization']))
                    crop = st.checkbox('Center crop / pad', value=p.get('crop_size') is not None)
                    if crop:
                        p['crop_size'] = [int(col.number_input(f'Crop {axis}', 1, max(1024, int(value)), int(value)))
                            for col, axis, value in zip(st.columns(3), ('R', 'A', 'S'), p.get('crop_size') or p['target_shape'])]
                    else:
                        p['crop_size'] = None
                    p['padding_mode'] = st.selectbox('Padding', ['constant', 'edge', 'reflect'], index=['constant', 'edge', 'reflect'].index(p.get('padding_mode', 'constant')))
                    resample = st.checkbox('Resample spacing in mm', value=p.get('target_spacing') is not None)
                    p['target_spacing'] = [float(col.number_input(f'Spacing {axis} (mm)', min(.1, float(value)), max(20., float(value)), float(value)))
                        for col, axis, value in zip(st.columns(3), ('R', 'A', 'S'), p.get('target_spacing') or [1., 1., 1.])] if resample else None
            preview_key = selected+json.dumps(p, sort_keys=True)
            if st.button('Run preprocessing preview', type='primary'):
                try:
                    validate_config({**config, 'preprocessing': p})
                    with st.spinner('Preprocessing this volume…'):
                        result, processed = preprocess_volume(volume, meta, p)
                    st.session_state.preview = (preview_key, result, processed)
                except Exception as exc:
                    st.error(str(exc) if isinstance(exc, ValueError) else f'Processing could not complete: {type(exc).__name__}')
            preview = st.session_state.get('preview')
            if preview and preview[0] == preview_key:
                result, processed = preview[1:]
                if page == 'Preprocessing':
                    plane = st.selectbox('Comparison plane', list(PLANES))
                    count = result.shape[PLANES[plane]]
                    index = st.slider('Synchronized slice', 0, count-1, count//2) if count > 1 else 0
                    aligned = aligned_original(volume, meta['affine'], processed, meta['spatial_unit'])
                    with st.container(key='image_grid'):
                        for col, array, title in zip(st.columns(2), (aligned, result), ('Original · aligned to output grid', 'Processed preview')):
                            with col:
                                fig = slice_figure(array, plane, index, processed['voxel_spacing'])
                                viewport(array, 'Acquired-series slice', 0, title, f'{array.shape} · range {array.min():.3f} to {array.max():.3f}', title, 400, figure=fig)
                    st.caption('Physical-grid-aligned comparison. Display contrast is independently scaled because normalization changes intensities; inspect cropped-away anatomy in MRI Viewer.')
                    with st.expander('Preview spatial metadata'):
                        st.json(processed)
                    buffer = io.BytesIO(); np.save(buffer, result[None], allow_pickle=False)
                    downloads: list[tuple[str, str | bytes, str]] = [
                        ('Download preview tensor', buffer.getvalue(), f'{selected}.npy'),
                        ('Download spatial metadata', json.dumps(processed, indent=2), f'{selected}.json'),
                        ('Download preview configuration', yaml.safe_dump({**config, 'preprocessing': p}), 'preview-config.yaml')]
                    for col, (label, content, name) in zip(st.columns(3), downloads):
                        col.download_button(label, content, name, width='stretch')
                with st.expander('Intensity distributions and numerical statistics', expanded=page == 'Intensity Analysis'):
                    for col, array, title in zip(st.columns(2), (volume, result), ('Original intensity', 'Processed intensity')):
                        with col:
                            st.plotly_chart(histogram(array, title), width='stretch')
                            st.dataframe(pd.DataFrame([intensity_stats(array)]), hide_index=True)
                if st.button('Calculate interpolation reconstruction baseline'):
                    try:
                        _, measured = downsample_reconstruction_baseline(result, float(result.max()-result.min()))
                        st.write(measured)
                        st.caption('Half-resolution downsample/linear upsample measures interpolation loss, not diagnostic quality.')
                    except ValueError as exc:
                        st.warning(str(exc))
            else:
                if page == 'Preprocessing':
                    st.caption('Ready to preview. Adjust optional parameters above, then run the preview to compare the same physical slices.')
                    plane = 'Sagittal'; index = canonical.shape[PLANES[plane]]//2
                    viewport(canonical, 'Acquired-series slice', 0, 'Current input', 'Canonical RAS · preview has not changed this scan', 'nifti_input', 360,
                             figure=slice_figure(canonical, plane, index, nib.affines.voxel_sizes(affine)))
                else:
                    st.plotly_chart(histogram(volume, 'Original intensity'), width='stretch')
                    st.dataframe(pd.DataFrame([intensity_stats(volume)]), hide_index=True, width='stretch')
                    st.caption('Run preprocessing preview to compare the intensity distribution after the configured operations.')
