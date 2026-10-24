# Finished project demo guide

## Launch

From the repository root, using the existing research-enabled environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open **http://127.0.0.1:8501**. The default page is **Project Results**.
Use `python -m pip install -e ".[dev,metrics,research]"` after installing the
appropriate PyTorch build when setting up another machine.

The published aggregate table is in Git. The real MRI data, checkpoints and saved
reconstructions stay in local `data/` and `runs/` folders. Without those local
artifacts, the app shows recorded benchmark numbers and an explicit unavailable-
images state rather than fabricated examples.

## Five-minute presentation

| Time | Show | Explain in plain language |
|---|---|---|
| 0:00–0:45 | Project Results title and metric cards | We compare how learned compressed representations reconstruct real knee MRI. The main demonstrated gain is reconstruction, not realistic new knee synthesis. |
| 0:45–2:15 | MRI comparison, default KL model, median case | All three panels show the same real exam and slice. Move the synchronized slider, then show a worst or seeded-random case. VQ is blurrier; KL retains more visible structure. |
| 2:15–2:45 | Optional interpolation and absolute-error maps | Interpolation is a simple non-learned reference. Error brightness uses a fixed scale. Pixel accuracy alone does not verify a ligament tear. |
| 2:45–3:30 | Experiment results | Fifteen runs, nine variants, equal 1,920-update/effective-batch-two budgets. Main VQ/iso-KL/aniso-KL comparisons have three seeds; other variants are one-seed screens. |
| 3:30–4:15 | Run the trained reconstruction now | Click **Reconstruct this MRI**. It runs the actual saved encoder/decoder on the selected input; it is not training or returning a preselected output. |
| 4:15–5:00 | Method & plain-language guide; research status | Explain the pipeline and the terms using the glossary. Describe the remaining generation/conditioning limits honestly, then download the report-writing handoff. |

**Optional:** select the higher-detail KL model and explain its larger latent
representation/no float-scalar compression. The headline cards remain explicitly
the three-seed anisotropic-KL benchmark, while the image captions report the
selected run's whole-volume metrics.

**Optional generation demonstration:** open **Model Experiments**, inspect saved
historical diffusion samples, and use its live generation controls if needed.
These are visibly experimental and are not the project's reconstruction result.
Avoid spending the main presentation waiting for diffusion sampling.

## Useful explanations

- **Reconstruction:** rebuilding an input MRI after compression.
- **Generation:** creating a new volume from noise.
- **Anisotropic compression:** keeping more positions in the acquired slice direction.
- **PSNR:** pixel-error score, higher is better.
- **SSIM:** structural-similarity score, higher is better.
- **Seed:** fixed random starting value for a repeatable experiment.
- **Conditioning:** requesting labels, not confirming pathology.

These explanations also appear in sidebar help, metric tooltips and the glossary.

## Timing and browser checks

The checked Chrome 156 dashboard had no observed JavaScript exceptions, console
errors or failed requests. The comparison, all three confirmation seeds,
best/median/worst/random navigation, interpolation, errors, CSV/report downloads,
preserved prototype navigation and 900/390 px layouts passed.

On the measured RTX 3060 Laptop host, first live reconstruction including model
loading took **28.46 seconds**, and a cached-model call took **0.49 seconds**.
These are observations, not latency guarantees. For a smooth presentation,
reconstruct the default case once beforehand. Viewing saved comparisons does
not load the trained model or start GPU inference.

Screenshots, the executable browser checks and machine-readable report are in
`runs/PROJECT-DEMO-20261024/`. See `docs/REPORT_HANDOFF.md` for the report outline,
figure captions, exact evidence locations and claims to use/avoid.
