# Local project walkthrough

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open **http://127.0.0.1:8501**. The navigation contains:

1. **Project overview:** the study settings and paired MRI preview.
2. **Reconstruction studio:** select a model/exam, move the linked slice slider,
   adjust the shared display window and add VQ/interpolation references.
3. **Experiment results:** inspect seed variation and download the measurement CSV.
4. **Knee dataset:** browse the independent acquired series and exam labels.
5. **Generation lab:** select a saved sample or submit a seeded inference request.
6. **About & terminology:** read the short workflow and searchable definitions.
7. **Data workbench:** inspect NIfTI volumes, preview preprocessing, render a 3D
   volume and export a separate processed dataset.

The reconstruction controls include image height, comparison references,
absolute-error maps, representative cases and PNG export. Live reconstruction
is stored for the selected model/case so slice changes reuse the resulting volume.

The generation controls include the seed, sampler, number of DDIM steps and
global label inputs. New requests create their own model-bound sample folders.

Data and model connection fields are in the relevant advanced sections. Local
MRI, checkpoint and reconstruction files are used for interactive views; the
aggregate study table is included in Git. Viewing saved MRI does not start
model training. Initial live inference loads the saved model; later calls reuse it.

Report-writing material is in `docs/REPORT_HANDOFF.md`, outside the website.

## Interface references

The workspace uses native page navigation, progressively disclosed settings,
linked image viewports and responsive layouts. Reference material:

- [Streamlit page/navigation](https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation)
- [Progressive disclosure, Nielsen Norman Group](https://www.nngroup.com/articles/progressive-disclosure/)
- [OHIF image viewports](https://docs.ohif.org/user-guide/viewer/viewport/)
- [W3C reflow guidance](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html)
- [W3C contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)

Research project only. Not for real medical use.
