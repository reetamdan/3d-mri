# M4 — Module Implementation and Trained-Model Demonstration

**Assessment window:** October 26–30, 2026 · **Weight:** 3 marks

## 1. Launch the trained product

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open http://127.0.0.1:8501. Use **MRNet Dataset** and **Model Experiments**.
The old Slicer brain scans remain a separate NIfTI integration-test dataset.

The new experiment is `runs/M4-MRNET-20261021`. Read
`runs/latest_m4.json` for the latest completed M4 run. Its computed submission
evidence is `evidence.md` and `evidence.json`; inspect these files before
presenting results. Their existence and verified content, not this guide alone,
establish that a training run completed.

## 2. Seven-minute demo sequence

| Time | Action | What it establishes |
|---|---|---|
| 0:00–0:45 | State the problem and show the architecture from M1/M2 | Problem, objective, method |
| 0:45–1:45 | Open MRNet Dataset; show source manifest, official split, real labels, and three acquired series | Real collected data and explicit data contract |
| 1:45–2:30 | Show processed counts, shape, normalization, and metadata checksums | Implemented data-handling design |
| 2:30–3:30 | Open Model Experiments; show VQ/diffusion curves, epochs, selected checkpoints, and optimizer steps | Actual training rather than unexecuted model code |
| 3:30–4:45 | Open Paired reconstruction results; compare a validation input and reconstruction; click **Reconstruct selected input now** | Trained encoder/codebook/decoder inference |
| 4:45–6:00 | Inspect saved generated volumes; open Live trained-model generation; change the seed and click **Generate a new volume** | Real random-noise diffusion and decoding |
| 6:00–7:00 | Show held-out metrics, diversity diagnostics, seeded repeatability, and limitations | Measured results and accurate interpretation |

The live sampler runs the checkpoint on available hardware; it does not return
a preselected saved image. New outputs and their manifests are stored beneath
the experiment's `interactive/` folder.

## 3. Reproduce from source data

Install base dependencies, then the GPU override on a compatible NVIDIA machine:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe scripts/download_mrnet_subset.py --train 96 --validation 24 --max-mib 1024
.\.venv\Scripts\python.exe scripts/run_m4.py
```

The explicit downloader fetches individual source files from the public
`cjinny/mrnet-v1` Kaggle mirror, version 1. It does not require a token for the
current public endpoint. The manifest records that Kaggle lists its license as
unknown; no redistribution license or official Stanford access approval is
asserted. Source data and weights are excluded from Git.

The runner preserves official splits, verifies source hashes, prepares sagittal
tensors, trains VQ and diffusion, evaluates held-out reconstruction, generates
samples, computes diagnostics, and exports evidence. Existing output folders
require an explicit resume:

```powershell
.\.venv\Scripts\python.exe scripts/run_m4.py --resume
```

Use new `--data` and `--output` folders for a different experiment. If the selected
VQ checkpoint changes after a latent cache was created, build a new cache/run;
do not attach old latents or generation reports to the new model.

## 4. Artifact map

```text
data/mrnet/MRNet-v1.0/
  source_manifest.json       # exact source, labels, selected exams, checksums
  train/{axial,sagittal,coronal}/*.npy
  valid/{axial,sagittal,coronal}/*.npy
data/processed/mrnet-m4-20261021/
  metadata.csv, experiment.json, report.json
runs/M4-MRNET-20261021/
  status.json               # environment, stages, measured runtime/memory
  vq/{last.pt,best.pt,history.json,run.json}
  latents/manifest.json
  diffusion/{last.pt,best.pt,history.json,run.json}
  reconstruction/report.json
  generated/manifest.json
  generation-report.json
  baseline-report.json
  seed-repeat/              # actual same-device repeated-seed check
  evidence.json, evidence.md
  evidence-assets/*.png
  interactive/              # new samples requested in the dashboard
```

## 5. Results and limitations to explain

- Report actual validation MSE/MAE/PSNR/SSIM from `evidence.json`.
- Explain whether the learned reconstruction outperforms the interpolation
  reference on each metric; do not omit an unfavorable comparison.
- Show actual generated samples and class conditioning. Conditions are requests,
  not verified pathology annotations for the generated outputs.
- Pairwise MS-SSIM describes similarity; high similarity can expose low diversity,
  and low similarity can reflect artifacts or noise.
- The selected validation split is not an independent hidden test set.
- This is reduced-cohort, reduced-resolution training. Expert anatomical review,
  full-cohort generalization, and downstream augmentation benefit remain separate
  studies. The demo establishes working trained modules.
- Inspection of the current generated samples shows blurred MRI-like texture
  and unreliable large-scale knee structure. Present this as a model limitation;
  the training and live inference are genuine, but realistic knee synthesis is
  not an achieved result. VQ reconstruction is substantially more interpretable.

## 6. Likely review questions

**Why latent diffusion?** It separates reconstruction from generative learning
and reduces spatial computation. Include the latent channel count when quoting
actual compression.

**Why VQ-VAE rather than claiming VQ-GAN results?** The trained baseline uses
stable paired reconstruction. The optional discriminator is implemented but not
silently claimed as part of a run with zero adversarial weight.

**How is leakage avoided?** Official MRNet splits are preserved; source and
processed metadata record their assignments. The released arrays do not expose
patient IDs, so the adapter relies on Stanford's published patient-disjoint policy.

**Why not Dice?** MRNet supplies exam labels, not aligned segmentation masks.
Reconstruction and generation use the appropriate available metrics.

**What is your contribution?** A reproducible, hardware-measured implementation,
auditable cohort and lineage, working 3D latent modeling, held-out comparisons,
and live trained inference. Scientific novelty or clinical utility is not inferred
from implementation alone.
