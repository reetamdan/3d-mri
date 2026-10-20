import numpy as np
import plotly.graph_objects as go


def intensity_stats(volume):
    return {'Minimum': float(volume.min()), 'Maximum': float(volume.max()),
            'Mean': float(volume.mean(dtype=np.float64)), 'Median': float(np.median(volume)),
            'Std deviation': float(volume.std(dtype=np.float64))}


def histogram(volume, title='Intensity distribution'):
    counts, edges = np.histogram(volume, bins=100)
    fig = go.Figure(go.Bar(x=(edges[1:]+edges[:-1])/2, y=counts, marker_color='#58cbbb'))
    fig.update_layout(title=title, template='plotly_dark', height=320, bargap=0,
                      xaxis_title='Intensity', yaxis_title='Voxel count', margin=dict(l=20, r=20, t=45, b=30))
    return fig
