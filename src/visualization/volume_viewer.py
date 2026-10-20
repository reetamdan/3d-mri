import numpy as np
import plotly.graph_objects as go


def volume_figure(volume, affine):
    # At most 48^3 values reach the browser.
    steps = np.maximum(np.ceil(np.array(volume.shape)/48).astype(int), 1)
    values = volume[tuple(slice(None, None, int(s)) for s in steps)]
    indices = np.indices(values.shape).reshape(3, -1) * steps[:, None]
    world = np.asarray(affine)[:3, :3] @ indices + np.asarray(affine)[:3, 3, None]
    lo, hi = np.percentile(values, [30, 99])
    fig = go.Figure(go.Volume(x=world[0], y=world[1], z=world[2], value=values.ravel(),
                              isomin=float(lo), isomax=float(hi), opacity=.12,
                              surface_count=12, colorscale='Gray', showscale=False))
    fig.update_layout(template='plotly_dark', height=560, margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(aspectmode='data', xaxis_title='R', yaxis_title='A', zaxis_title='S'))
    return fig
