"""MRNet import and research artifact inspection, separate from the NIfTI dashboard."""
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.data.mrnet import TASKS, PLANES, labels_for, load_series, prepare_mrnet
from src.visualization.mri_viewer import slice_figure
from src.visualization.plots import histogram
from src.utils.integrity import confined_path, sha256_file, generated_volume
from src.utils.artifacts import resolve_reference


def research_runs(root):
    """Only index completed, traceable research checkpoints; never promote smoke artifacts."""
    result = []
    for path in sorted(Path(root).glob('*/vq/run.json'), reverse=True):
        try:
            info = json.loads(path.read_text())
            if not isinstance(info, dict):
                continue
            epochs = info.get('completed_epochs', 0)
            checkpoint_file = info.get('checkpoint_file', 'last.pt')
            if (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1
                    or not isinstance(checkpoint_file, str)):
                continue
            checkpoint = confined_path(path.parent, checkpoint_file)
            if info.get('smoke_test', True):
                continue
            if checkpoint.is_file() and sha256_file(checkpoint) == info.get('checkpoint_sha256'):
                result.append(path.parent.parent)
        except (OSError, ValueError):
            continue
    return result


def mrnet_page(root):
    st.info('MRNet knee data and the separate Slicer brain test samples use different data contracts. Source manifests record any downloaded subset.')
    st.link_button('MRNet access at Stanford', 'https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c')
    folder = Path(st.text_input('Extracted MRNet folder', str(root/'data/mrnet/MRNet-v1.0')))
    st.caption('Expected: train/{axial,sagittal,coronal}/*.npy, valid/... and six headerless label CSVs.')
    if not folder.is_dir():
        st.warning('MRNet files are not present at this location. Provide an extracted dataset or run the explicit subset downloader.')
        return
    try:
        provenance = folder/'source_manifest.json'
        if provenance.exists():
            source = json.loads(provenance.read_text())
            st.caption(f'Source: {source["source_url"]} · completed subset: {source["complete"]}')
            with st.expander('Acquisition and cohort provenance'):
                st.json({k: v for k, v in source.items() if k != 'files'})
        split = st.selectbox('Official split', ['train', 'valid'])
        labels = labels_for(folder, split)
        st.metric('Labeled exams', len(labels))
        exams = sorted(labels)
        if not exams:
            st.warning('No labeled exams found.')
            return
        index = st.selectbox('Exam', list(range(len(exams))), format_func=lambda i: f'Exam {i+1:04d}')
        exam = exams[index]
        for col, plane in zip(st.columns(3), PLANES):
            with col:
                path = folder/split/plane/f'{exam}.npy'
                if not path.exists():
                    st.warning(f'Missing {plane} series')
                    continue
                volume = load_series(path)
                count = volume.shape[0]
                i = st.slider(f'{plane.title()} acquired slice', 0, count-1, count//2) if count > 1 else 0
                fig = go.Figure(go.Heatmap(z=volume[i], colorscale='Gray', showscale=False))
                fig.update_layout(title=f'{plane.title()} · {i+1}/{count}', height=350, template='plotly_dark')
                fig.update_yaxes(autorange='reversed', scaleanchor='x')
                st.plotly_chart(fig, use_container_width=True)
        st.caption('These are independently acquired series. Slice positions are not synchronized across planes; physical affine/spacing are absent in MRNet arrays.')
        st.write(dict(zip(TASKS, labels[exam])))
        st.subheader('Prepare a model dataset')
        plane = st.selectbox('Training series', PLANES, index=1)
        target = Path(st.text_input('New processed run folder', str(root/'data/processed/mrnet-sagittal')))
        limit = st.number_input('Maximum exams per official split', 1, 2000, 20)
        shape = [int(col.number_input(f'Target {axis}', min_value=8, max_value=512, value=value, step=8))
                 for col, axis, value in zip(st.columns(3), ('rows', 'columns', 'slices'), (128, 128, 32))]
        if st.button('Validate & prepare MRNet subset', type='primary'):
            with st.spinner(f'Validating MRNet and resizing to {shape}…'):
                report = prepare_mrnet(folder, target, plane, shape=shape, limit=int(limit))
            if report['valid_scans']:
                st.success(f'Prepared {report["valid_scans"]} scans')
            else:
                st.error('No scans were exported. Review validation_errors.csv in the output directory.')
            st.json(report)
    except Exception as exc:
        st.error(f'MRNet processing failed: {type(exc).__name__}. Check dataset layout, labels, and output directory.')


def experiments_page(root):
    st.caption('Only completed research runs with verified checkpoint checksums appear here. Test artifacts are excluded.')
    st.markdown('**Pipeline:** MRI → VQ encoder/codebook → latent DDPM → VQ decoder → evaluation → augmentation comparison')
    st.caption('MRNet classification uses exam labels. Segmentation Dice requires separately supplied aligned masks.')
    runs_dir = Path(st.text_input('Research runs directory', str(root/'runs')))
    folders = research_runs(runs_dir)
    if not folders:
        st.info('No verified research checkpoints are available. Prepare a real dataset with train/validation splits and train a model before inspecting results here.')
        st.code('python scripts/research.py train-vq --data <processed-dataset> --output runs/<experiment>/vq --config configs/models.yaml', language='text')
        return
    run = st.selectbox('Experiment folder', folders, format_func=lambda p: p.name)
    vq_info = json.loads((run/'vq/run.json').read_text())
    vq_checkpoint = confined_path(run/'vq', vq_info.get('checkpoint_file', 'last.pt'))
    st.warning('Completed training and matching checksums do not establish anatomical validity. Independent evaluation and qualified anatomical review are still required.')
    evidence_path = run/'evidence.json'
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        st.caption(evidence['status'])
        cols = st.columns(4)
        cols[0].metric('Training / validation exams', f'{evidence["data"]["train_count"]} / {evidence["data"]["validation_count"]}')
        cols[1].metric('Held-out VQ SSIM', f'{evidence["reconstruction"]["ssim"]:.4f}')
        cols[2].metric('Held-out VQ PSNR', f'{evidence["reconstruction"]["psnr"]:.2f} dB')
        cols[3].metric('VQ / diffusion updates', f'{evidence["vq"]["optimizer_steps"]} / {evidence["diffusion"]["optimizer_steps"]}')
        st.download_button('Download M3/M4 evidence', evidence_path.read_bytes(), 'evidence.json')
    summary_path = run/'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        st.caption(summary.get('status', 'Research run'))
    cols = st.columns(2)
    for col, stage, y in zip(cols, ['vq', 'diffusion'], ['reconstruction_l1', 'noise_mse']):
        with col:
            st.subheader(f'{stage.upper()} training')
            history = run/stage/'history.json'
            if history.exists():
                frame = pd.DataFrame(json.loads(history.read_text()))
                st.line_chart(frame.set_index('epoch')[[y]])
                st.dataframe(frame, hide_index=True)
            else:
                st.info('No training history for this stage')
    manifest_path = run/'generated/manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        diffusion_info = json.loads((run/'diffusion/run.json').read_text())
        diffusion_checkpoint = confined_path(run/'diffusion', diffusion_info.get('checkpoint_file', 'last.pt'))
        if (manifest['vq_sha256'] != sha256_file(vq_checkpoint)
                or manifest['diffusion_sha256'] != sha256_file(diffusion_checkpoint)):
            raise ValueError('Generated samples do not match the selected model checkpoints')
        st.subheader('Generated volumes')
        if manifest['smoke_test']:
            st.error('This generated output is test-only and cannot be displayed as a research result.')
            return
        selected = st.selectbox('Generated sample', [r['file'] for r in manifest['rows']])
        sample_path = confined_path(run/'generated', selected)
        row = next(r for r in manifest['rows'] if r['file'] == selected)
        if sha256_file(sample_path) != row.get('sha256'):
            st.error('Generated tensor integrity cannot be verified. Regenerate this artifact.')
            return
        data = generated_volume(run/'generated', row, manifest['shape'])
        st.caption(f'Tensor shape: {data.shape}. Display uses tensor-axis slices; no patient-space affine is assigned.')
        section = st.selectbox('Tensor slice plane', ['Acquired-series slice', 'Row cross-section', 'Column cross-section'])
        plane, axis = {'Acquired-series slice': ('Axial', 2), 'Row cross-section': ('Sagittal', 0),
                       'Column cross-section': ('Coronal', 1)}[section]
        idx = st.slider('Generated slice', 0, data.shape[axis+1]-1, data.shape[axis+1]//2)
        fig = slice_figure(data[0], plane, idx)
        fig.update_layout(title=f'{section} · {idx+1}/{data.shape[axis+1]}', xaxis_title='Tensor axis', yaxis_title='Tensor axis')
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(histogram(data, 'Generated intensity distribution'), use_container_width=True)
        st.download_button('Download generated tensor', (run/'generated'/selected).read_bytes(), selected)
        with st.expander('Live trained-model generation'):
            st.caption('Runs the selected diffusion and VQ checkpoints now. Each request writes a fresh, traceable output folder.')
            seed = int(st.number_input('Sampling seed', min_value=0, max_value=2147483647, value=2026))
            sampling = manifest.get('sampling', {})
            method = st.selectbox('Sampling method', ['ddpm', 'ddim'], index=int(sampling.get('method') == 'ddim'))
            training_steps = int(sampling.get('training_steps', 200))
            sampling_steps = int(st.number_input('DDIM sampling steps', 1, training_steps, min(50, training_steps))) if method == 'ddim' else None
            guidance_trained = sampling.get('classifier_free_guidance_trained', sampling.get('guidance_scale', 1.) != 1.)
            guidance = float(st.number_input('Classifier-free guidance scale', 0., 5., 1.5, .1)) if guidance_trained else 1.
            condition = manifest['rows'][0].get('conditioning_labels')
            labels = [int(col.checkbox(task.title(), value=bool(value)))
                      for col, task, value in zip(st.columns(3), TASKS, condition)] if condition is not None else None
            if st.button('Generate a new volume', type='primary'):
                from datetime import datetime
                import uuid
                from src.training.experiments import generate
                target = run/'interactive'/('sample-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
                with st.spinner('Sampling Gaussian noise with the trained model…'):
                    generate(vq_checkpoint, diffusion_checkpoint, target, count=1, seed=seed, labels=labels,
                             sampler=method, sampling_steps=sampling_steps, guidance_scale=guidance)
                st.session_state['live_generation'] = dict(run=str(run), folder=str(target))
            live = st.session_state.get('live_generation')
            if live and live['run'] == str(run):
                folder = Path(live['folder'])
                record = json.loads((folder/'manifest.json').read_text())['rows'][0]
                array = generated_volume(folder, record, manifest['shape'])[0]
                figure = slice_figure(array, 'Axial', array.shape[2]//2, limits=(-1, 1))
                figure.update_layout(title=f'New model sample · seed {record["seed"]}', xaxis_title='Tensor column', yaxis_title='Tensor row')
                st.plotly_chart(figure, use_container_width=True)
                st.success('New inference completed and checksum-verified.')
                st.download_button('Download new generated volume', (folder/record['file']).read_bytes(), record['file'])
    for title, relative in [('Paired reconstruction results', 'reconstruction/report.json'),
                             ('Generation diagnostics', 'generation-report.json'),
                             ('Augmentation comparison', 'downstream/report.json')]:
        path = run/relative
        with st.expander(title):
            if path.exists():
                report = json.loads(path.read_text())
                if report.get('smoke_test', True):
                    st.info('Test-only or legacy report excluded from research results.')
                    continue
                st.json(report, expanded=False)
                st.download_button(f'Download {title.lower()}', path.read_bytes(), path.name, key=relative)
                if relative == 'reconstruction/report.json':
                    from src.training.common import TrainingDataset, fingerprint
                    input_dataset = resolve_reference(path.parent, report['input_dataset'])
                    if report['checkpoint_sha256'] != sha256_file(vq_checkpoint) or report['dataset_fingerprint'] != fingerprint(input_dataset):
                        raise ValueError('Reconstruction report does not match the current checkpoint or dataset; evaluate again')
                    shape = report['shape']
                    source = TrainingDataset(input_dataset, report['split'], shape)
                    scan_id = st.selectbox('Reconstruction pair', [r['scan_id'] for r in report['rows']])
                    index = next(i for i, r in enumerate(source.rows) if r['scan_id'] == scan_id)
                    original = source[index][0][0].numpy()
                    pair = next(r for r in report['rows'] if r['scan_id'] == scan_id)
                    restored = generated_volume(run/'reconstruction', pair, shape)[0]
                    idx = st.slider('Paired tensor slice', 0, original.shape[2]-1, original.shape[2]//2)
                    for col, volume, label in zip(st.columns(2), [original, restored], ['Input tensor', 'VQ reconstruction']):
                        with col:
                            st.caption(label)
                            fig = slice_figure(volume, 'Axial', idx, limits=(-1, 1))
                            fig.update_layout(title=f'Acquired-series slice · {idx+1}/{original.shape[2]}',
                                              xaxis_title='Tensor axis', yaxis_title='Tensor axis')
                            st.plotly_chart(fig, use_container_width=True)
                    if st.button('Reconstruct selected input now'):
                        import torch
                        from src.training.common import device_for
                        from src.training.experiments import load_vq
                        from src.metrics import reconstruction_metrics
                        device = device_for()
                        with st.spinner('Running the trained encoder, codebook and decoder…'):
                            model, _ = load_vq(vq_checkpoint, device)
                            with torch.no_grad():
                                result = model(source[index][0][None].to(device))[0][0, 0].cpu().numpy()
                        st.write(reconstruction_metrics(original, result))
                        figure = slice_figure(result, 'Axial', idx, limits=(-1, 1))
                        figure.update_layout(title='Live trained VQ reconstruction', xaxis_title='Tensor column', yaxis_title='Tensor row')
                        st.plotly_chart(figure, use_container_width=True)
                        st.success('Live reconstruction inference completed.')
            else:
                st.info('No results recorded. Do not infer successful completion from available model code.')
    st.markdown('See **docs/milestones-2-5.md** for reproducible training commands, dataset requirements, and research limitations.')

    tokenizer = run/'tokenizer.json'
    if tokenizer.exists():
        with st.expander('Tokenizer usage and quantization diagnostics'):
            report = json.loads(tokenizer.read_text())
            st.write({k: report[k] for k in ('samples', 'used_codes', 'dead_code_fraction', 'aggregate_perplexity', 'quantization_mse')})
            st.bar_chart(report['counts'])
