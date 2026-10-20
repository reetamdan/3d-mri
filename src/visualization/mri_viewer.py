import numpy as np
import nibabel as nib
import plotly.graph_objects as go

PLANES = {'Axial': 2, 'Sagittal': 0, 'Coronal': 1}


def canonical_volume(volume, affine):
    image = nib.as_closest_canonical(nib.Nifti1Image(volume, np.asarray(affine)))
    return np.asarray(image.dataobj), image.affine


def slice_figure(volume, plane, index, spacing=(1, 1, 1), limits=None):
    axis = PLANES[plane]
    data = np.take(volume, index, axis=axis).T
    remaining = [i for i in range(3) if i != axis]
    labels = ['R →', 'A →', 'S →']
    lo, hi = limits if limits else (float(volume.min()), float(volume.max()))
    fig = go.Figure(go.Heatmap(z=data, colorscale='Gray', zmin=lo, zmax=hi, showscale=False,
                              dx=float(spacing[remaining[0]]), dy=float(spacing[remaining[1]])))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=20), template='plotly_dark',
                      title=f'{plane} · {index+1}/{volume.shape[axis]}',
                      xaxis_title=labels[remaining[0]], yaxis_title=labels[remaining[1]])
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig


def aligned_original(volume, original_affine, processed_metadata, original_unit):
    """Sample original intensities on the processed grid for spatially synchronized comparison."""
    from nibabel.processing import resample_from_to
    affine = np.array(original_affine, dtype=float)
    if processed_metadata['spatial_unit'] == 'mm' and original_unit in ('meter', 'micron'):
        affine[:3, :] *= {'meter': 1000, 'micron': .001}[original_unit]
    source = nib.Nifti1Image(volume, affine)
    img = resample_from_to(source, (tuple(processed_metadata['volume_shape']), np.array(processed_metadata['affine'])), order=1)
    return np.asarray(img.dataobj)
