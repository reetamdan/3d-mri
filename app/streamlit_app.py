"""Local research dashboard. Source file names and subject identifiers stay out of views."""
import io
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from src.utils.config import load_config, validate_config
from src.data.pipeline import inspect_dataset, run_pipeline
from src.data.loader import load_mri
from src.data.preprocessing import preprocess_volume
from src.data.splitting import split_patients
from src.visualization.mri_viewer import PLANES, canonical_volume, slice_figure, aligned_original
from src.visualization.plots import histogram, intensity_stats
from src.visualization.volume_viewer import volume_figure
from src.metrics import downsample_reconstruction_baseline
from mri3d.configuration import workspace_root

ROOT = workspace_root()
st.set_page_config(page_title='3D MRI Research', page_icon='◈', layout='wide')
st.markdown('''<style>
.block-container {padding-top:4.5rem; max-width:1500px}
h1 {font-weight:650; letter-spacing:-1.2px}
[data-testid="stMetric"] {background:#16212d;border:1px solid #283848;padding:18px;border-radius:10px}
[data-testid="stSidebar"] {border-right:1px solid #283848}
@media (max-width:1100px) {
  .block-container {padding-left:1rem; padding-right:1rem}
  .block-container [data-testid="stHorizontalBlock"] {flex-direction:column}
  .block-container [data-testid="stColumn"] {width:100% !important; flex:1 1 auto !important; min-width:0 !important}
}
</style>''', unsafe_allow_html=True)
st.sidebar.markdown('## ◈ MRI RESEARCH')
st.sidebar.caption('KNEE MRI / DATA · RECONSTRUCTION · DIFFUSION')
page = st.sidebar.radio('Workspace', ['Project Results', 'Dataset Overview', 'MRI Viewer', 'Preprocessing', 'Intensity Analysis', 'Dataset Quality', 'Export', 'MRNet Dataset', 'Model Experiments'])
with st.sidebar.expander('Technical terms, simply explained'):
    from app.glossary import TERMS
    for term in ('Reconstruction', 'Generation', 'Latent representation', 'VQ-VAE', 'KL autoencoder', 'PSNR', 'SSIM', 'Conditioning'):
        st.markdown(f'**{term}** — {TERMS[term]}')
if page == 'Project Results':
    from app.project_results import project_results_page
    st.caption('3D MEDICAL IMAGING / PROJECT RESULTS')
    st.title('3D Knee MRI')
    project_results_page(ROOT)
    st.sidebar.caption('Research prototype · measured results, not clinical diagnosis.')
    st.stop()
config_path = st.sidebar.text_input('Configuration file', str(ROOT/'configs/config.yaml')
                                  if (ROOT/'configs/config.yaml').is_file() else '',
                                  help='Leave blank for the bundled preprocessing preset in the current workspace.')
try:
    config = load_config(config_path)
except Exception as exc:
    st.error(f'Configuration could not be loaded: {type(exc).__name__}. Check the YAML and paths.')
    st.stop()
raw_dir = st.sidebar.text_input('Raw dataset directory', config['dataset']['raw_dir'])
config['dataset']['raw_dir'] = str(Path(raw_dir).resolve())
fingerprint = json.dumps(config, sort_keys=True)
if st.session_state.get('dataset_key') != fingerprint:
    for key in ('records', 'errors', 'report', 'preview', 'run'):
        st.session_state.pop(key, None)
    st.session_state.dataset_key = fingerprint
st.sidebar.caption('NIfTI .nii / .nii.gz · local processing')
if st.sidebar.button('↻ Index & validate dataset', type='primary', width='stretch'):
    # Clear the previous snapshot before indexing, including when indexing fails.
    for key in ('records', 'errors', 'report', 'preview', 'run'):
        st.session_state.pop(key, None)
    try:
        with st.spinner('Reading and validating scans…'):
            records, errors, indexed_report = inspect_dataset(config)
        st.session_state.update(records=records, errors=errors, report=indexed_report)
        st.session_state.pop('preview', None)
        st.session_state.pop('run', None)
    except Exception as exc:
        st.error(str(exc) if isinstance(exc, ValueError) else f'Dataset indexing failed: {type(exc).__name__}')
st.sidebar.divider()
st.sidebar.caption('Research prototype — not for clinical diagnosis or treatment.')
st.caption('RESEARCH WORKSPACE     /     DATA FOUNDATION')
st.title(page)
st.write('Inspect, standardize, and prepare 3D volumes for latent representation learning.')
if page in ('MRNet Dataset', 'Model Experiments'):
    from app.research_workspace import mrnet_page, experiments_page
    if page == 'MRNet Dataset':
        mrnet_page(ROOT)
    else:
        try:
            experiments_page(ROOT)
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            st.error(f'The selected research artifacts cannot be read: {type(exc).__name__}. Check the run files and provenance.')
    st.stop()
records = st.session_state.get('records', [])
report = st.session_state.get('report')
if report is None:
    st.info('Connect your dataset to begin. Set the raw directory in the sidebar, then select “Index & validate dataset”.')
    st.markdown('### A reproducible path from volume to tensor')
    cols = st.columns(3)
    for col, title, detail in zip(cols, ['01 / Inspect', '02 / Standardize', '03 / Export'],
        ['Validate NIfTI volumes and review spatial metadata.', 'Orient to RAS, resample, crop or pad, and normalize.', 'Create subject-safe splits with complete processing provenance.']):
        with col:
            st.subheader(title)
            st.write(detail)
    st.code('data/raw/\n  subject_001/\n    scan.nii.gz\n  subject_002/\n    scan.nii.gz', language='text')
    st.caption('No data is downloaded automatically. If local samples are already present, click Index & validate dataset. Flat directories require a CSV manifest with file_path and patient_id columns.')
    st.stop()

if page == 'Dataset Overview':
    splits = split_patients([r['patient_id'] for r in records], config['split'], config['seed'])
    cols = st.columns(4)
    cols[0].metric('Valid scans', len(records))
    for col, split in zip(cols[1:], ('train', 'validation', 'test')):
        col.metric(f'{split.title()} · planned', sum(splits[r['patient_id']] == split for r in records))
    st.caption('Planned counts use currently valid subjects. Export assigns all indexed subjects before processing; final counts are recorded in the run report.')
    st.subheader('Dataset geometry')
    if records:
        st.dataframe(pd.DataFrame({'Statistic': ['Minimum', 'Mean', 'Maximum'],
                     'Shape · R/A/S order only after canonicalization': [str(report['shape'][k]) for k in ('min', 'mean', 'max')],
                     'Spacing · native units': [str(report['spacing_native_units'][k]) for k in ('min', 'mean', 'max')]}), hide_index=True, width='stretch')
        st.caption(f'{report["subjects"]} subjects · {report["distinct_shapes"]} distinct shapes · units: {report["spatial_units"]}')
    st.metric('Invalid scans', report['invalid_scans'])
    st.caption('Scan and subject labels are pseudonymous hashes, not source names. Hashing is not full anonymization.')

elif page == 'Dataset Quality':
    cols = st.columns(3)
    for col, key in zip(cols, ('discovered_scans', 'valid_scans', 'invalid_scans')):
        col.metric(key.replace('_', ' ').title(), report[key])
    if st.session_state.get('errors'):
        st.dataframe(st.session_state.errors, hide_index=True, width='stretch')
    else:
        st.success('No invalid scans detected in the current index.')
    st.json(report, expanded=True)
    st.download_button('Download quality report', json.dumps(report, indent=2), 'quality_report.json', 'application/json')

elif page == 'Export':
    st.subheader('Create a processed dataset')
    st.write('Exports use the YAML configuration. Preview controls do not change batch settings. Each run creates a new directory with tensors, metadata, configuration, and a quality report.')
    st.code(yaml.safe_dump(config['preprocessing']), language='yaml')
    st.caption(f'Format: {config.get("export_format", "npy")} · Tensor: [1, R, A, S] · Seed: {config["seed"]}')
    if st.button('Process & export dataset', type='primary', disabled=not records):
        bar = st.progress(0.)
        try:
            st.session_state.run = run_pipeline(config, progress=bar.progress)
        except Exception as exc:
            st.error(f'Export failed: {type(exc).__name__}. Check output permissions and configuration.')
    if st.session_state.get('run'):
        run = st.session_state.run
        message = f'Exported {len(run["rows"])} scans; {len(run["errors"])} rejected.'
        (st.success if run['rows'] else st.error)(message)
        st.code(run['output_dir'], language='text')
        st.json(run['report'])
        for name in ('metadata.csv', 'report.json', 'validation_errors.csv'):
            st.download_button(f'Download {name}', (Path(run['output_dir']) / name).read_bytes(), name)

else:
    if not records:
        st.warning('No valid scans available. Review Dataset Quality or choose another dataset.')
        st.stop()
    selected = st.selectbox('Scan', [r['scan_id'] for r in records])
    record = next(r for r in records if r['scan_id'] == selected)
    try:
        volume, meta = load_mri(record['path'], config['preprocessing'].get('max_voxels', 67108864))
    except Exception:
        st.error('This scan is no longer readable. Re-index the dataset.')
        st.stop()
    canonical, affine = canonical_volume(volume, meta['affine'])
    if page == 'MRI Viewer':
        cols = st.columns(3)
        for col, plane in zip(cols, PLANES):
            with col:
                count = canonical.shape[PLANES[plane]]
                idx = st.slider(f'{plane} slice', 0, count-1, count//2) if count > 1 else 0
                st.plotly_chart(slice_figure(canonical, plane, idx, nib.affines.voxel_sizes(affine)), width='stretch')
        st.caption(f'Original shape: {meta["volume_shape"]} · Original orientation: {meta["orientation"]} · Display: canonical RAS · Units: {meta["spatial_unit"]}')
        with st.expander('Spatial metadata'):
            st.json({k: v for k, v in meta.items() if k != 'file_path'})
        if st.checkbox('Render 3D volume (downsampled for performance)'):
            st.plotly_chart(volume_figure(canonical, affine), width='stretch')
    else:
        p = dict(config['preprocessing'])
        if page == 'Preprocessing':
            with st.expander('Preview parameters', expanded=True):
                st.caption('Preview changes apply here only. Download the YAML to reuse them for batch export.')
                shape_cols = st.columns(3)
                p['target_shape'] = [int(col.number_input(f'Target {label}', min_value=1, max_value=max(1024, int(v)), value=int(v))) for col, label, v in zip(shape_cols, ('R', 'A', 'S'), p['target_shape'])]
                p['normalization'] = st.selectbox('Normalization', ['minmax', 'zscore'], index=['minmax', 'zscore'].index(p['normalization']))
                crop = st.checkbox('Center crop / pad', value=p.get('crop_size') is not None)
                if crop:
                    cols = st.columns(3)
                    p['crop_size'] = [int(col.number_input(f'Crop {axis}', 1, max(1024, int(v)), int(v))) for col, axis, v in zip(cols, ('R', 'A', 'S'), p.get('crop_size') or p['target_shape'])]
                else:
                    p['crop_size'] = None
                p['padding_mode'] = st.selectbox('Padding', ['constant', 'edge', 'reflect'], index=['constant', 'edge', 'reflect'].index(p.get('padding_mode', 'constant')))
                resample = st.checkbox('Resample spacing in mm', value=p.get('target_spacing') is not None)
                if resample:
                    cols = st.columns(3)
                    p['target_spacing'] = [float(col.number_input(f'Spacing {axis} (mm)', min(.1, float(v)), max(20., float(v)), float(v))) for col, axis, v in zip(cols, ('R', 'A', 'S'), p.get('target_spacing') or [1., 1., 1.])]
                else:
                    p['target_spacing'] = None
        preview_key = selected + json.dumps(p, sort_keys=True)
        if st.button('Run preprocessing preview', type='primary'):
            try:
                validate_config({**config, 'preprocessing': p})
                with st.spinner('Preprocessing selected volume…'):
                    result, processed_meta = preprocess_volume(volume, meta, p)
                st.session_state.preview = (preview_key, result, processed_meta)
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else f'Processing failed: {type(exc).__name__}')
        preview = st.session_state.get('preview')
        if preview and preview[0] == preview_key:
            result, processed_meta = preview[1:]
            if page == 'Preprocessing':
                plane = st.selectbox('Comparison plane', list(PLANES))
                count = result.shape[PLANES[plane]]
                idx = st.slider('Synchronized slice', 0, count-1, count//2) if count > 1 else 0
                aligned = aligned_original(volume, meta['affine'], processed_meta, meta['spatial_unit'])
                cols = st.columns(2)
                for col, data, title in zip(cols, (aligned, result), ('Original intensities · aligned to output grid', 'Processed')):
                    with col:
                        st.subheader(title)
                        st.plotly_chart(slice_figure(data, plane, idx, processed_meta['voxel_spacing']), width='stretch')
                        st.caption(f'Shape {data.shape} · Range {data.min():.3f} to {data.max():.3f}')
                st.caption(f'Source shape: {volume.shape}. Comparison samples the original on the output world-coordinate grid; cropped-away anatomy must be checked in MRI Viewer. Display contrast is scaled independently.')
                st.json(processed_meta, expanded=False)
                buffer = io.BytesIO()
                np.save(buffer, result[None], allow_pickle=False)
                st.download_button('Download preview tensor', buffer.getvalue(), f'{selected}.npy')
                st.download_button('Download spatial metadata', json.dumps(processed_meta, indent=2), f'{selected}.json')
                st.download_button('Download preview configuration', yaml.safe_dump({**config, 'preprocessing': p}), 'preview-config.yaml')
            cols = st.columns(2)
            for col, data, title in zip(cols, (volume, result), ('Original intensity', 'Processed intensity')):
                with col:
                    st.plotly_chart(histogram(data, title), width='stretch')
                    st.dataframe(pd.DataFrame([intensity_stats(data)]), hide_index=True)
            if st.button('Calculate interpolation reconstruction baseline'):
                try:
                    _, metrics = downsample_reconstruction_baseline(result, float(result.max()-result.min()))
                    st.write(metrics)
                    st.caption('Half-resolution downsampling followed by linear reconstruction on the same grid. This measures interpolation loss, not diagnostic quality.')
                except ValueError as exc:
                    st.warning(str(exc))
        else:
            st.plotly_chart(histogram(volume, 'Original intensity'), width='stretch')
            st.info('Run a preview to compare the original and processed volumes.')
