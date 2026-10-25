"""Shared presentation styling and faithful image viewports for the research app."""
import base64
from html import escape
import io
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from src.visualization.mri_viewer import tensor_slice_figure, TENSOR_SECTIONS

CSS = '''<style>
:root {--lab-ink:#ecf2fa;--lab-muted:#a4b5cd;--lab-line:#26364b;--lab-panel:#111d2f;--lab-mint:#62e0c2}
.stApp {background:radial-gradient(ellipse at 100% 0%,#13243a 0%,#0a1220 48%)}
[data-testid="stAppHeader"] {background:#0a1220dd}
[data-testid="stAppDeployButton"] {display:none}
.block-container {max-width:1540px;padding:2.7rem 2.3rem 3rem}
h1 {font-size:2.2rem!important;font-weight:680!important;letter-spacing:-1.2px;line-height:1.13!important;padding-top:.15rem!important;padding-bottom:.35rem!important}
h2 {font-size:1.35rem!important;font-weight:650!important;letter-spacing:-.35px}
h3 {font-size:1.05rem!important}
p,li {line-height:1.6}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p {color:var(--lab-muted)!important;font-size:.86rem}
[data-testid="stSidebar"] {background:#0b1422;border-right:1px solid var(--lab-line)}
[data-testid="stSidebar"][data-collapsed="false"] {min-width:242px!important;max-width:242px!important}
[data-testid="stSidebar"][data-collapsed="true"] {min-width:0!important;max-width:0!important}
[data-testid="stSidebarContent"] {padding-top:1.1rem}
[data-testid="stSidebar"] [data-testid="stPageLink"] a {min-height:43px;padding:.65rem .8rem;border-radius:9px;font-size:.92rem}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {background:#17283c}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {background:#173a3c;color:#8bf4da;border:1px solid #2d625b}
[data-testid="stSidebar"] [data-testid="stPageLink"] {margin-bottom:2px}
[data-testid="stMetric"] {background:var(--lab-panel);border:1px solid var(--lab-line);padding:15px;border-radius:12px;min-height:106px}
[data-testid="stMetricLabel"] {color:var(--lab-muted)}
[data-testid="stMetricValue"] {font-size:1.9rem}
button:focus-visible,a:focus-visible,input:focus-visible {outline:2px solid var(--lab-mint)!important;outline-offset:3px!important}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button {min-height:40px;border-radius:8px;font-weight:550}
button[kind="primary"],button[kind="primaryFormSubmit"],button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primaryFormSubmit"] {color:#06251e!important;font-weight:650!important}
[data-testid="stExpander"] {border-color:var(--lab-line);border-radius:10px}
[data-testid="stDataFrame"] {border:1px solid var(--lab-line);border-radius:10px}
.lab-brand {display:flex;gap:10px;align-items:center;margin:0 0 20px;color:var(--lab-ink)}
.lab-mark {display:grid;place-items:center;width:40px;height:40px;border-radius:12px;background:#173b3c;color:#80f1d5;font-size:26px;border:1px solid #2a625b}
.lab-brand strong {letter-spacing:1.5px;font-size:14px}.lab-brand small {display:block;color:var(--lab-muted);font-size:11px;margin-top:3px;letter-spacing:.5px}
.lab-nav-label {font-size:10px!important;color:#8ea4c1;letter-spacing:1.7px;margin:20px 0 7px!important;font-weight:700}
.lab-eyebrow {display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;font-size:10px;letter-spacing:1.7px;color:#a8bbd4;font-weight:650;margin:0 0 4px}
.lab-badge {display:inline-flex;align-items:center;gap:6px;border:1px solid #315b57;background:#15352f;color:#93f2d9;padding:5px 10px;border-radius:30px;letter-spacing:.25px;font-size:11px;font-weight:600}
.lab-badge.amber {background:#332b18;border-color:#6b5830;color:#f3d997}
.lab-badge.neutral {background:#162338;border-color:#33475f;color:#b8c9df}
.lab-stats {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0 17px}
.lab-stat {background:linear-gradient(135deg,#14243a,#101b2c);border:1px solid var(--lab-line);padding:15px 17px;border-radius:12px;min-width:0}
.lab-stat small {display:block;color:#b1c2d7;font-size:11px;margin-bottom:7px}
.lab-stat strong {font-size:27px;letter-spacing:-.7px;font-weight:640;color:var(--lab-ink)}
.lab-stat span {display:block;color:#a9bdd4;font-size:11px;margin-top:5px;line-height:1.5}
.lab-stat .positive {color:#8be7cb}
.lab-note {background:#111e30;border-left:3px solid #52cfb3;padding:12px 16px;border-radius:0 8px 8px 0;color:#b9c9dd;font-size:13px;line-height:1.6;margin:10px 0 16px}
.lab-note.amber {border-color:#d9b96c}
.lab-preview {display:grid;grid-template-columns:1fr 1fr;gap:8px;background:#050a12;padding:10px;border:1px solid #2c4058;border-radius:14px}
.lab-preview figure {margin:0;min-width:0}.lab-preview img {width:100%;aspect-ratio:1;object-fit:contain;display:block;border-radius:7px;background:black}
.lab-preview figcaption {color:#b7c9df;font-size:11px;padding:8px 2px 0}
.lab-route-card {height:100%;background:#101d30;border:1px solid var(--lab-line);border-radius:12px;padding:18px;margin-bottom:8px}
.lab-route-card small {color:#73dfc3;font-size:10px;letter-spacing:1.4px;font-weight:700}
.lab-route-card h3 {margin:10px 0 7px!important;color:#eff5fc}.lab-route-card p {font-size:13px;color:#aebed4;margin:0;line-height:1.6}
.lab-pipeline {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}
.lab-step {border:1px solid #2a3c54;border-radius:10px;padding:15px;background:#111e30}.lab-step b {display:block;color:#dce7f6;font-size:13px;margin:8px 0}.lab-step small {color:#9cafc7;font-size:12px}
[class*="st-key-viewport_"] {background:#050a12!important;border-color:#2a3b51!important;padding:.65rem!important;border-radius:12px!important;gap:.35rem!important}
[class*="st-key-viewport_"] [data-testid="stVerticalBlock"] {gap:.35rem}
[class*="st-key-viewport_"] [data-testid="stCaptionContainer"] {font-size:.75rem}
.st-key-studio_toolbar [data-testid="stVerticalBlock"],.st-key-view_controls [data-testid="stVerticalBlock"] {gap:.4rem}
[class*="st-key-page_heading_"] {gap:.2rem!important}
[class*="st-key-page_heading_"] [data-testid="stVerticalBlock"] {gap:.2rem}
.st-key-studio_canvas {gap:.5rem!important}
@media(max-width:900px){.block-container {padding:2.5rem 1.1rem 2rem}.lab-stats {grid-template-columns:repeat(2,minmax(0,1fr))}.lab-pipeline {grid-template-columns:repeat(2,minmax(0,1fr))}h1 {font-size:1.9rem!important}}
@media(max-width:650px){.lab-eyebrow {font-size:9px;letter-spacing:1px}.lab-badge {font-size:10px;padding:4px 7px}.lab-stat {padding:12px}.lab-stat strong {font-size:24px}.lab-pipeline {grid-template-columns:1fr}
[data-testid="stMain"] [data-testid="stHorizontalBlock"] {flex-direction:column}
[data-testid="stMain"] [data-testid="stColumn"] {width:100%!important;flex:1 1 100%!important;min-width:0!important}
.st-key-image_grid [data-testid="stHorizontalBlock"],.st-key-studio_toolbar [data-testid="stHorizontalBlock"],.st-key-view_controls [data-testid="stHorizontalBlock"] {flex-direction:column}
.st-key-image_grid [data-testid="stColumn"],.st-key-studio_toolbar [data-testid="stColumn"],.st-key-view_controls [data-testid="stColumn"] {width:100%!important;flex:1 1 100%!important;min-width:0!important}}
@media print{[data-testid="stSidebar"],[data-testid="stAppHeader"] {display:none}.block-container {padding:0}}
</style>'''


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(section: str, title: str, description: str, status: str = '', tone: str = 'neutral') -> None:
    badge = f'<span class="lab-badge {escape(tone)}">{escape(status)}</span>' if status else ''
    with st.container(key='page_heading_'+re.sub(r'[^a-zA-Z0-9]', '_', title)):
        st.markdown(f'<div class="lab-eyebrow"><span>KNEE MRI LAB / {escape(section.upper())}</span>{badge}</div>', unsafe_allow_html=True)
        st.title(title)
        st.caption(description)


def stat_cards(rows: list[tuple[str, str, str]]) -> None:
    cards = ''.join(f'<div class="lab-stat"><small>{escape(label)}</small><strong>{escape(value)}</strong><span>{escape(detail)}</span></div>'
                    for label, value, detail in rows)
    st.markdown(f'<div class="lab-stats">{cards}</div>', unsafe_allow_html=True)


def note(text: str, tone: str = '') -> None:
    st.markdown(f'<div class="lab-note {escape(tone)}">{escape(text)}</div>', unsafe_allow_html=True)


def display_png(plane: np.ndarray, limits: tuple[float, float] = (-1., 1.)) -> bytes:
    low, high = limits
    if not np.isfinite([low, high]).all() or high <= low or plane.ndim != 2:
        raise ValueError('Display requires a 2D image and increasing finite contrast limits')
    pixels = np.rint(np.clip((plane-low)/(high-low), 0, 1)*255).astype(np.uint8)
    buffer = io.BytesIO(); Image.fromarray(pixels).save(buffer, format='PNG')
    return buffer.getvalue()


def preview_pair(original: np.ndarray, reconstruction: np.ndarray, index: int) -> None:
    figures = []
    for array, label in ((original, 'REAL VALIDATION MRI'), (reconstruction, 'TRAINED KL RECONSTRUCTION')):
        encoded = base64.b64encode(display_png(array[:, :, index].T)).decode('ascii')
        figures.append(f'<figure><img alt="{label}" src="data:image/png;base64,{encoded}"><figcaption>{label}</figcaption></figure>')
    st.markdown('<div class="lab-preview">'+''.join(figures)+'</div>', unsafe_allow_html=True)


def viewport(array: np.ndarray, section: str, index: int, label: str, detail: str, key: str,
             height: int = 390, limits: tuple[float, float] = (-1., 1.), figure: Any = None) -> None:
    with st.container(border=True, key='viewport_'+re.sub(r'[^a-zA-Z0-9_-]', '_', key)):
        st.markdown(f'**{label}**', help='Scroll to zoom, drag to inspect, and double-click to reset. Display changes do not alter the image data.')
        st.caption(detail)
        fig = figure if figure is not None else tensor_slice_figure(array, section, index, limits=limits)
        fig.update_layout(height=height, title=None, margin=dict(l=5, r=5, t=5, b=5),
            paper_bgcolor='#050a12', plot_bgcolor='#050a12', uirevision=key, dragmode='zoom')
        fig.update_xaxes(visible=False, showgrid=False)
        fig.update_yaxes(visible=False, showgrid=False)
        st.plotly_chart(fig, width='stretch', key=key,
            config={'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})


def comparison_figure(arrays: list[np.ndarray], labels: list[str], section: str, index: int,
                      limits: tuple[float, float], caption: str) -> bytes:
    """Export the displayed planes at a shared contrast; never sharpen or alter model arrays."""
    width, gap, top = 320, 16, 40
    canvas = Image.new('RGB', (len(arrays)*(width+gap)+gap, width+top+55), '#0a1220')
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(size=14)
    for i, (array, title) in enumerate(zip(arrays, labels)):
        image = Image.open(io.BytesIO(display_png(np.take(array, index, axis=TENSOR_SECTIONS[section]).T, limits))).convert('RGB')
        image.thumbnail((width, width), Image.Resampling.NEAREST)
        # thumbnail does not upscale; explicit nearest-neighbor display enlargement
        # keeps the original sample lattice visible in the downloadable figure.
        factor = min(width/image.width, width/image.height)
        image = image.resize((round(image.width*factor), round(image.height*factor)), Image.Resampling.NEAREST)
        x = gap+i*(width+gap)
        draw.text((x, 12), title, fill='#e9f1fa', font=font)
        canvas.paste(image, (x+(width-image.width)//2, top+(width-image.height)//2))
    draw.text((gap, width+top+12), caption, fill='#a8bdd3', font=ImageFont.load_default(size=12))
    output = io.BytesIO(); canvas.save(output, format='PNG'); return output.getvalue()
