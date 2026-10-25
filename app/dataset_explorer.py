"""Presentation of the actual acquired knee series, separate from optional NIfTI tools."""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.artifact_cache import read_json, verified_hash, file_signature, FileSignature
from app.ui import page_header, stat_cards, viewport, note
from src.data.mrnet import labels_for, load_series, prepare_mrnet, PLANES, TASKS
from src.utils.integrity import confined_path


@st.cache_data(show_spinner=False, max_entries=6)
def _series(signature: FileSignature, expected: str) -> np.ndarray:
    if expected and verified_hash(signature[0]) != expected:
        raise ValueError('Acquired MRI checksum mismatch')
    array = load_series(signature[0])
    if file_signature(signature[0]) != signature:
        raise ValueError('Acquired MRI changed while loading')
    return array


def dataset_page(root: Path) -> None:
    page_header('The source data', 'Knee dataset', 'Explore the independently acquired MRI series and real exam labels used in this study.', 'MRNet · acquired images')
    with st.expander('Dataset connection & source provenance'):
        folder = Path(st.text_input('Extracted MRNet folder', str(root/'data/mrnet/MRNet-v1.0')))
        st.link_button('MRNet access at Stanford', 'https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c')
        st.caption('Expected train/ and valid/ folders, three acquired series and the official label CSVs.')
    if not folder.is_dir():
        note('Local MRNet images are not connected. Choose the extracted dataset above; the recorded research results remain available in Experiment results.')
        return
    try:
        provenance = read_json(folder/'source_manifest.json') if (folder/'source_manifest.json').is_file() else {}
        training, validation = labels_for(folder, 'train'), labels_for(folder, 'valid')
        stat_cards([('Training exams', str(len(training)), 'Official source training assignment'),
                    ('Validation exams', str(len(validation)), 'Used for model selection'),
                    ('Acquired series', '3', 'Axial · sagittal · coronal'),
                    ('Identity scope', 'Exam level', 'Patient linkage is not supplied')])
        controls = st.columns([1, 2])
        split = controls[0].selectbox('Official split', ['train', 'valid'], format_func=lambda value: 'Training' if value == 'train' else 'Validation')
        labels = training if split == 'train' else validation
        exams = sorted(labels)
        if not exams:
            st.info('No labeled exams were found in this split.'); return
        chosen = controls[1].selectbox('Exam', range(len(exams)), format_func=lambda i: f'Exam {i+1:03d} / {len(exams)}')
        exam = exams[chosen]
        hashes = {r['file'].replace('\\', '/'): r['sha256'] for r in provenance.get('files', [])}
        with st.container(key='image_grid'):
            for col, plane in zip(st.columns(3), PLANES):
                with col:
                    relative = f'{split}/{plane}/{exam}.npy'
                    path = confined_path(folder, relative)
                    if not path.is_file():
                        st.info(f'No {plane} acquisition is available for this exam.'); continue
                    array = _series(file_signature(path), hashes.get(relative, ''))
                    index = st.slider(f'{plane.title()} acquired slice', 1, len(array), len(array)//2+1, key=f'acquired_slice:{split}:{exam}:{plane}')
                    figure = go.Figure(go.Heatmap(z=array[index-1], colorscale='Gray', zmin=0, zmax=255, showscale=False))
                    figure.update_yaxes(autorange='reversed', scaleanchor='x')
                    viewport(array, 'Acquired-series slice', 0, f'{plane.title()} acquisition',
                             f'Slice {index} / {len(array)} · source array {array.shape[1]} × {array.shape[2]}',
                             f'raw_{plane}', 315, figure=figure)
        label_cards = [(f'{task.title()} label', 'Positive' if value else 'Negative', 'Source exam-level annotation')
                       for task, value in zip(TASKS, labels[exam])]
        stat_cards(label_cards+[('Scope', 'Exam level', 'Binary dataset labels')])
        st.caption('These are separate acquired series, not synchronized reslices. MRNet arrays do not supply a physical affine, verified laterality or millimeter spacing.')
        with st.expander('Cohort balance & acquisition details'):
            positive = np.asarray(list(training.values())).sum(0)
            figure = go.Figure(go.Bar(x=list(TASKS), y=positive, marker_color='#62e0c2'))
            figure.update_layout(height=260, yaxis_title='Positive training exams', margin=dict(t=10, l=15, r=15, b=20))
            st.plotly_chart(figure, width='stretch')
            if provenance:
                st.json({key: value for key, value in provenance.items() if key != 'files'}, expanded=False)
        with st.expander('Prepare a separate model dataset'):
            st.caption('Advanced operation: creates a new processed cohort. It does not change the frozen 96/24 study.')
            plane = st.selectbox('Training series', PLANES, index=1)
            target = Path(st.text_input('New processed run folder', str(root/'data/processed/mrnet-sagittal')))
            limit = st.number_input('Maximum exams per official split', 1, 2000, 20)
            shape = [int(col.number_input(f'Target {axis}', min_value=8, max_value=512, value=value, step=8))
                     for col, axis, value in zip(st.columns(3), ('rows', 'columns', 'slices'), (128, 128, 32))]
            if st.button('Validate & prepare MRNet subset', type='primary'):
                with st.spinner('Validating and exporting the separate cohort…'):
                    result = prepare_mrnet(folder, target, plane, shape=shape, limit=int(limit))
                st.success(f'Prepared {result["valid_scans"]} scans') if result['valid_scans'] else st.error('No scans were exported; review the validation report.')
                st.json(result, expanded=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'The acquired dataset could not be read ({type(exc).__name__}). Check the connection, source arrays and labels.')
