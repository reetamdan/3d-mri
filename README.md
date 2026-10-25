# 3D Knee MRI Research

A local research project for MRI exploration, learned reconstruction and conditional
latent diffusion, implemented with PyTorch, MONAI and a Streamlit workspace.

[![Tests](https://github.com/reetamdan/3d-mri/actions/workflows/tests.yml/badge.svg)](https://github.com/reetamdan/3d-mri/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-3D%20models-EE4C2C)

## Workspace

- **Project overview:** study configuration, recorded measurements and paired MRI previews.
- **Reconstruction studio:** linked slices, shared contrast, VQ/KL models, reference
  images, error maps, live reconstruction and comparison-figure download.
- **Experiment results:** model settings, training-seed variation and CSV export.
- **Knee dataset:** acquired MRNet series and exam-level labels.
- **Generation lab:** saved samples and seeded DDPM/DDIM inference.
- **About & terminology:** a short workflow description and one-line explanations.
- **Data workbench:** NIfTI inspection, preprocessing, 3D viewing and export.

Reports and presentation notes are maintained in `docs/`, outside the website navigation.

## Quick start

Use Python 3.12. From a terminal:

```bash
git clone https://github.com/reetamdan/3d-mri.git
cd 3d-mri
python -m venv .venv
```

Activate on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install a CPU PyTorch build and the project:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,metrics,research]"
python -m streamlit run app/streamlit_app.py
```

Open **http://127.0.0.1:8501**. Recorded aggregate measurements are included in
the repository. Interactive MRI tools read the matching local datasets and run
artifacts. Data and weights are stored in ignored `data/` and `runs/` directories.

For NVIDIA CUDA:

```bash
python -m pip install -r requirements-cuda.txt
python -m pip install -e ".[dev,metrics,research]"
```

The recorded constrained-hardware study used an RTX 3060 Laptop with 6 GiB VRAM,
torch 2.14.1+cu130 and MONAI 1.6.1. `requirements-research.txt` records direct
research-package pins; other installation variants are described by the
`requirements-*.txt` files.

## Research workflow

```text
MRI → preprocessing → VQ or KL representation → reconstruction
                                  ↓
                         frozen latent cache
                                  ↓
                    conditional latent diffusion → sampled volume
```

The project includes anisotropic KL sampling, L1/MS-SSIM/gradient objectives,
explicit frozen medical features, training-fitted latent scaling, Min-SNR,
EMA, classifier-free guidance and condition-token attention configurations.

## Recorded reconstruction study

Same 96 training / 24 validation exams, 128×128×32 inputs, 1,920 optimizer updates
per run and effective batch two. Values are means ± sample SD across seeds 42–44.

| Representation | PSNR dB | SSIM |
|---|---:|---:|
| Matched VQ baseline | 24.352 ± 0.100 | 0.7691 ± 0.0081 |
| KL isotropic | 25.502 ± 0.070 | 0.8169 ± 0.0037 |
| KL anisotropic | 26.802 ± 0.199 | 0.8578 ± 0.0092 |

The [measured table](docs/results/ae_bakeoff_measured.md) lists all 15 runs and
nine representation variants. [The research report](docs/research_v2_report.md)
records the experimental methods, sampling studies and classification measurements.

## Data and experiment commands

```bash
python scripts/download_mrnet_subset.py --train 96 --validation 24 --max-mib 1024
python scripts/run_m4.py
python scripts/representation_study.py --output runs/AE-BAKEOFF-new --device cuda
```

The MRNet adapter retains official exam assignments and records source/processed
content hashes. The NIfTI pipeline uses supplied spatial metadata and subject
grouping. Source datasets have their own access and use terms; the repository
contains application code and aggregate experimental measurements.

For a NIfTI processing workflow:

```bash
python scripts/validate_dataset.py
python scripts/preprocess.py
python scripts/generate_report.py
```

The installed `mri3d` CLI exposes data preparation, training, caching, sampling,
evaluation and artifact commands. Public YAML presets in `configs/` are also
included in the installed package.

## Documentation

- [Report-writing handoff](docs/REPORT_HANDOFF.md)
- [Research methods and measurements](docs/research_v2_report.md)
- [External repositories, source classes and papers](docs/external_repos_review.md)
- [Architecture and baseline identifiers](docs/current_baseline.md)
- [Data-pipeline reference](docs/data-pipeline.md)
- [Model configuration card](docs/model-card.md)
- [Local walkthrough](docs/PROJECT_DEMO.md)
- [Academic milestone material](docs/milestones/README.md)

## Development

```bash
python -m ruff check .
python -m mypy
python -m pytest -q --cov --cov-report=term-missing
python -m pip check
```

The CI workflow runs CPU checks on Linux and Windows, including package/preset
installation and a 70% statement/branch coverage threshold. Dataset- and
CUDA-dependent cases use the corresponding local resources. No test downloads
MRI data automatically. See [SECURITY.md](SECURITY.md) for credential handling.

Research project only. Not for real medical use.
