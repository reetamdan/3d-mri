"""Native multipage entrypoint: a guided presentation with a separate data workbench."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from app.ui import apply_theme

st.set_page_config(page_title='Knee MRI Lab', page_icon='◈', layout='wide')
apply_theme()
presentation = [
    st.Page('pages/overview.py', title='Project overview', icon=':material/space_dashboard:', default=True),
    st.Page('pages/reconstruction.py', title='Reconstruction studio', icon=':material/compare:'),
    st.Page('pages/experiments.py', title='Experiment results', icon=':material/monitoring:'),
    st.Page('pages/dataset.py', title='Knee dataset', icon=':material/layers:'),
    st.Page('pages/generation.py', title='Generation lab', icon=':material/blur_on:'),
    st.Page('pages/about.py', title='About & terminology', icon=':material/help_outline:'),
]
workbench = st.Page('pages/workbench.py', title='Data workbench', icon=':material/tune:')
selected = st.navigation(presentation+[workbench], position='hidden')
with st.sidebar:
    st.markdown('<div class="lab-brand"><span class="lab-mark">◈</span><div><strong>KNEE MRI LAB</strong><small>VOLUMETRIC IMAGING RESEARCH</small></div></div>', unsafe_allow_html=True)
    st.markdown('<p class="lab-nav-label">THE PROJECT</p>', unsafe_allow_html=True)
    for page in presentation:
        st.page_link(page)
    st.markdown('<p class="lab-nav-label">RESEARCH TOOLS</p>', unsafe_allow_html=True)
    st.page_link(workbench)
    st.divider()
    st.caption('A reproducible study of real knee MRI, learned reconstruction and experimental generation.')
    with st.expander('Quick terminology'):
        from app.glossary import TERMS
        for term in ('Reconstruction', 'Generation', 'PSNR', 'SSIM'):
            st.markdown(f'**{term}** — {TERMS[term]}')
    st.caption('Local research workspace')
st.set_page_config(page_title=f'{selected.title} · Knee MRI Lab')
selected.run()
st.divider()
st.caption('Research project only. Not for real medical use.')
