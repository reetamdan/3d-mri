# Report-writing handoff

## Project summary

The project studies 3D knee MRI preprocessing, representation learning,
reconstruction and conditional latent diffusion. The application provides MRI
viewers, synchronized VQ/KL comparisons, experiment measurements, sampling tools
and geometry-aware NIfTI processing. It uses PyTorch, MONAI and Streamlit.

## Suggested report chapters

1. **Introduction:** MRI representation learning and the project objectives.
2. **Related work:** VQ/KL representations, latent diffusion, conditioning and MRI
   augmentation. Source classes, papers and licenses are in `external_repos_review.md`.
3. **Dataset:** the official 96/24 MRNet exam assignments, acquired-series layout,
   resizing, normalization and available metadata.
4. **Methodology:** isotropic/anisotropic sampling, reconstruction objectives,
   matched optimizer budgets, checkpoint selection and training-fitted latent scaling.
5. **Implementation:** data modules, model classes, native application pages,
   caching, content hashes, CLI and configuration presets.
6. **Experiments:** the recorded representation, diffusion, conditioning and
   classification tables in `research_v2_report.md`.
7. **Discussion:** describe measured differences between settings in the context
   of this dataset, training budget and evaluation protocol.
8. **Conclusion:** summarize the implemented workflow and possible further studies.

## Main representation table

Means ± sample SD across seeds 42–44, with 1,920 updates per run and batch two:

| Configuration | PSNR dB | SSIM |
|---|---:|---:|
| Matched native VQ | 24.352 ± .100 | .7691 ± .0081 |
| KL isotropic | 25.502 ± .070 | .8169 ± .0037 |
| KL anisotropic | 26.802 ± .199 | .8578 ± .0092 |

The historical M4 VQ checkpoint is a different run: PSNR 24.29968 / SSIM .76587.
Its interpolation reference is 21.36067 / .77707. Keep those comparisons distinct.
All 15 representation runs are listed in `results/ae_bakeoff_measured.md`.
The diffusion and classification measurements are recorded in `research_v2_report.md`.

## Study scope and available components

- Recorded: data audit, 15 representation runs, selected latent statistics,
  bounded diffusion/conditioning experiments, medical-feature neighborhoods and
  the historical three-seed classification probe.
- Available implementations: local external-knee-data adapters, same-architecture
  initialization, mask-based measurements, balanced conditional sampling and a
  TorchIO/downstream comparison framework.
- Additional pretraining, spatial conditioning and independent-cohort experiments
  are possible separate studies; they are not included in the recorded tables.
- The 24 validation exams are used for model selection. Patient linkage and
  physical metadata are not supplied in the released MRNet arrays.

## Figures

| Figure | Source |
|---|---|
| Workflow | About & terminology page; `current_baseline.md` |
| Original / VQ / KL comparisons | Reconstruction studio's PNG download; local `runs/AE-BAKEOFF/comparison-*.png` |
| Representative cases | Baseline-median, baseline-lowest/highest PSNR and seeded-random case controls |
| Absolute errors | Reconstruction studio, fixed 0–.5 error range |
| Three-seed comparison | Experiment results charts and full-results CSV |
| Source acquisitions | Knee dataset page with the same selected exam |
| Sampled volumes | Generation lab, including the recorded seed and condition inputs |

Use identical slice positions and display windows in comparison figures. Record
the selected model, seed and case. Captions should identify reconstruction and
sampling as separate operations and specify whether scores are per-volume means
or across training seeds.

## Application and source map

- `app/streamlit_app.py`: native page router and shared footer.
- `app/ui.py`: visual styling, image viewports and comparison PNG export.
- `app/presentation.py`: overview, reconstruction, measurements and terminology.
- `app/dataset_explorer.py`: acquired-series exploration and cohort preparation.
- `app/generation_lab.py`: model-bound sample viewing and seeded inference.
- `app/workbench.py`: NIfTI inspection, preview, 3D viewing and export.
- `src/models/`, `src/training/`: model and research implementations.

Reports and presentation notes remain in repository documentation, outside the
website navigation. The website contains functional tools and terminology help.

## Reproduction and version references

Launch with `python -m streamlit run app/streamlit_app.py`. Setup is in the README;
experiment commands are in the research report and the public YAML configurations.
Recorded experimental metadata stays with local datasets, weights and run folders.

- Source: <https://github.com/reetamdan/3d-mri>.
- Current checks: <https://github.com/reetamdan/3d-mri/actions/workflows/tests.yml>.
- Exact revision: `git rev-parse origin/main`.
- The full functional run before this wording/cleanup pass completed 193 tests,
  73.71% statement/branch coverage and mypy across 96 source files. Current CI
  provides the source revision's corresponding check output.

Research project only. Not for real medical use.
